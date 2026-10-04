# oaqjp-final-project-emb-ai

A small Flask web application that scores a piece of customer feedback into five
emotions — **anger**, **disgust**, **fear**, **joy** and **sadness** — and reports the
dominant one.

The primary backend is the IBM Watson NLP emotion model
`emotion_aggregated-workflow_lang_en_stock`. Because that model is hosted on an IBM
SkillsBuild lab network that is frequently unreachable, the project ships a fully
local fallback (`SamLowe/roberta-base-go_emotions`) so the application keeps working
offline. Both backends emit the same five-label score distribution, so callers never
need to know which engine answered.

## Features

- Scores feedback into five emotions and returns the dominant emotion.
- Two interchangeable backends: the deployed Watson workflow, or a local GoEmotions
  transformer, selected by the `EMOTION_ENGINE` environment variable.
- Automatic fallback when the Watson workflow is down, with a failure cooldown so a
  dead endpoint is not retried on every keystroke.
- Aggregates GoEmotions' 28 fine-grained labels down onto the five Watson labels.
- A JSON health endpoint that reports which backends the process has configured.
- A JSON error envelope for every failure mode, with meaningful HTTP status codes.
- 77 unit tests that never download a model, plus 5 opt-in end-to-end checks.
- Type annotated, ships a PEP 561 `py.typed` marker, and scores 10.00/10 on pylint.

## How it works

```
browser ──► /emotionDetector ──► validate_text() ──► _resolve_scores()
                                                          │
                                    ┌─────────────────────┴─────────────────────┐
                                    ▼                                           ▼
                          _watson_scores()                          _local_scores()
                       POST to the Watson NLP                  GoEmotions transformer
                       emotion workflow                            (28 labels)
                                    │                                           │
                                    └──────────────┬────────────────────────────┘
                                                   ▼
                            aggregate + renormalise onto the five labels
                                                   ▼
                                    dominant_emotion = argmax(scores)
```

1. **`validate_text()`** rejects anything that is not a non-empty string of at most
   1000 characters, raising `InvalidInputError` (HTTP 400).
2. **`_resolve_scores()`** asks the Watson workflow first. On success those scores are
   returned as-is. On any failure — DNS error, timeout, non-200 status, malformed body,
   or missing field — the endpoint is put into a 5 minute cooldown and the local
   backend is used instead.
3. **`_local_scores()`** runs `SamLowe/roberta-base-go_emotions`, a _multi-label_
   classifier whose independent sigmoid outputs do not sum to one. `_aggregate_goemotions()`
   sums the 28 labels into their five Watson parents (for example `annoyance` and
   `disapproval` both fold into `anger`) and renormalises so the scores sum to 1.0.
4. `server.py` formats the scores into the sentence the browser displays.

## Requirements

- Python 3.10 or newer.
- No API key is needed. The Watson lab endpoint is unauthenticated, and the local
  backend downloads its weights from HuggingFace on first use.

> **Note:** the local backend needs `torch` and `transformers`, which is a large
> install (roughly 800 MB once installed, dominated by `torch`) plus a one-off ~480 MB
> model download. If you only want to serve the Watson workflow, install `Flask` and
> `requests` and set `EMOTION_ENGINE=watson`.

## Installation

With `make`:

```bash
make install        # runtime dependencies
make install-dev    # runtime + pytest, pylint, build
```

Or directly:

```bash
python -m pip install -r requirements-dev.txt
```

> The `Makefile` recipes assume a POSIX shell. On Windows, install
> [GnuWin32 Make](https://gnuwin32.sourceforge.net/packages/make.html) or run the
> raw `python` commands shown in each section below — they work in PowerShell too.

## Running the application

```bash
make run
# or
python server.py
```

Then open <http://localhost:5000>.

For development, use the debug reloader instead:

```bash
make dev
# or
python -m flask --app server run --debug
```

## Configuration

| Variable                    | Values                    | Default | Effect                                                                                                                                                                                                                                                             |
| --------------------------- | ------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `EMOTION_ENGINE`            | `auto`, `watson`, `local` | `auto`  | Which backend to use. `auto` prefers Watson and falls back to local. `watson` and `local` are strict: if the chosen backend is unavailable the request fails with HTTP 503 instead of falling back. An unrecognised value logs a warning and is treated as `auto`. |
| `EMOTION_INTEGRATION_TESTS` | `1`, `true`, `yes`        | unset   | Enables the 5 end-to-end emotion checks that need a real model.                                                                                                                                                                                                    |

```bash
# Force the local backend and skip the Watson call entirely
EMOTION_ENGINE=local python server.py
```

## HTTP API

### `GET /`

Renders the single page interface.

### `GET|POST /emotionDetector`

Scores the `textToAnalyse` field. The field is read from the JSON body first, then a
form-encoded body, then the query string, so all of these work:

```bash
curl "http://localhost:5000/emotionDetector?textToAnalyse=I+am+glad+this+happened"
curl -X POST -H "Content-Type: application/json" \
     -d '{"textToAnalyse":"I am glad this happened"}' \
     http://localhost:5000/emotionDetector
curl -X POST --data-urlencode "textToAnalyse=I am glad this happened" \
     http://localhost:5000/emotionDetector
```

**200 OK** — the formatted breakdown:

```
'anger' : 0.02, 'disgust' : 0.00, 'fear' : 0.01, 'joy' : 0.97 and 'sadness' : 0.01. The dominant emotion is joy.
```

**4xx / 5xx** — a JSON error envelope:

```json
{
  "error": {
    "code": "invalid_input",
    "message": "'textToAnalyse' must not be blank.",
    "status": 400
  }
}
```

### `GET /health`

Reports the configured backends without contacting any of them.

```bash
curl http://localhost:5000/health
```

```json
{
  "status": "ok",
  "backends": {
    "configured_engine": "auto",
    "watson_endpoint": "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict",
    "watson_model_id": "emotion_aggregated-workflow_lang_en_stock",
    "watson_in_cooldown": false,
    "local_model_id": "SamLowe/roberta-base-go_emotions",
    "local_backend_installed": true,
    "emotion_labels": ["anger", "disgust", "fear", "joy", "sadness"]
  }
}
```

### Error codes

| Status | `code`               | Raised when                                                                               |
| ------ | -------------------- | ----------------------------------------------------------------------------------------- |
| 400    | `invalid_input`      | `textToAnalyse` is missing, blank, not a string, or longer than 1000 characters.          |
| 404    | `not_found`          | The requested route does not exist.                                                       |
| 405    | `method_not_allowed` | The HTTP method is not supported on that route.                                           |
| 500    | `model_unavailable`  | A non-`InvalidInputError` emotion failure reached the error handler.                      |
| 500    | `internal_error`     | An unexpected server-side exception.                                                      |
| 503    | `model_unavailable`  | No emotion backend could answer, e.g. `EMOTION_ENGINE=watson` while the endpoint is down. |

## Testing

```bash
make test           # or: python -m pytest -q
make test-cov       # or: python -m pytest --cov=EmotionDetection --cov=server --cov-report=term-missing
```

The default suite is **77 passing, 5 skipped** and never downloads a model: both
backends are replaced with stubs, the Watson failure cooldown is reset around every
test, and the memoised local pipeline cache is cleared around every test. Measured
statement coverage of `EmotionDetection` and `server` is **98%** — the only uncovered
lines are the `transformers` `ImportError` fallback and the `if __name__ == "__main__"`
block, neither of which is reachable from a test.

### End-to-end emotion checks

`TestCanonicalEmotions` asserts the dominant emotion for five canonical sentences
against a _real_ model, so it is opt-in and skipped by default:

```bash
# POSIX shell
EMOTION_INTEGRATION_TESTS=1 python -m pytest -q -m integration
# or
make test-integration

# PowerShell
$env:EMOTION_INTEGRATION_TESTS=1; python -m pytest -q -m integration
```

| Feedback                                   | Expected dominant emotion |
| ------------------------------------------ | ------------------------- |
| `I am glad this happened`                  | `joy`                     |
| `I am really mad about this`               | `anger`                   |
| `I feel disgusted just hearing about this` | `disgust`                 |
| `I am so sad about this`                   | `sadness`                 |
| `I am really afraid that this will happen` | `fear`                    |

## Static code analysis

```bash
make lint
# or
python -m pylint server.py EmotionDetection test_emotion_detection.py
```

The project is currently rated **10.00/10**. The configuration lives in
`pyproject.toml` rather than a `.pylintrc`:

- `[tool.pylint.main]` — `py-version = "3.10"`, `jobs = 1` for reproducible output.
- `[tool.pylint.format]` — `max-line-length = 100`.
- `[tool.pylint.design]` — tightened limits: `max-args = 5`, `max-locals = 15`,
  `max-branches = 12`, `max-statements = 50`.
- `[tool.pylint.similarities]` — `min-similarity-lines = 8` to suppress noisy
  `duplicate-code` between the JSON and HTML request handlers.
- `[tool.pylint."messages control"]` — disables `duplicate-code` and
  `too-few-public-methods`.

Every module and public function carries a docstring; the package's custom exceptions
document the HTTP status code they carry.

## Packaging

```bash
make build
```

Produces a wheel and a source distribution via `setuptools`. `MANIFEST.in` includes the
templates, static assets, requirements files, `README.md` and `LICENSE`.

## Project layout

```
.
├── EmotionDetection/
│   ├── __init__.py             # public re-exports
│   ├── emotion_detection.py    # validation, both backends, aggregation
│   └── py.typed                # PEP 561 marker
├── static/
│   ├── mywebscript.js          # fetch client and result rendering
│   └── styles.css              # result panel styling
├── templates/
│   └── index.html              # single page interface
├── test_emotion_detection.py   # 77 unit tests + 5 opt-in integration tests
├── server.py                   # Flask routes and error handling
├── Makefile                    # install / test / lint / build / run
├── MANIFEST.in                 # sdist contents
├── pyproject.toml              # packaging, pytest and pylint configuration
├── requirements.txt            # runtime dependencies
└── requirements-dev.txt        # runtime + development dependencies
```

## Known limitations

- **The Watson endpoint is often unreachable.** `sn-watson-emotion.labs.skills.network`
  is an IBM SkillsBuild lab service, not a production SLA-backed endpoint. On an
  arbitrary network the connection typically times out, `auto` silently switches to
  the local backend, and `/health` will show `"watson_in_cooldown": true` for 5 minutes
  after the first failure. Set `EMOTION_ENGINE=local` to skip the Watson attempt and
  its timeout entirely.
- **The local backend needs a one-off ~480 MB download** from HuggingFace on first use,
  after which it is cached and memoised for the life of the process.
- **The GoEmotions aggregation is lossy.** Folding 28 labels into 5 is a heuristic, so
  the local scores are close to, but not identical with, the Watson model's. The two
  backends agree on the dominant emotion for ordinary single-emotion sentences.
- **Watson only accepts up to 1000 characters** of input, which is enforced before any
  backend is called.
- `EMOTION_INTEGRATION_TESTS` skips itself when no backend answers, so a green
  integration run is the only proof the models actually work; check the skip reason.

## License

Apache-2.0. See [LICENSE](LICENSE).
