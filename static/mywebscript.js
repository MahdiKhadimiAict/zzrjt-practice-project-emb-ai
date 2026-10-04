/**
 * Client for the Watson NLP emotion detector.
 *
 * Posts the feedback text to /emotionDetector and renders either the formatted
 * emotion breakdown (HTTP 200) or the JSON error envelope that server.py
 * returns for blank input (HTTP 400) and an unreachable model (HTTP 503).
 */

const ENDPOINT = "/emotionDetector";

/**
 * Render a message inside the result panel.
 *
 * @param {string} html - Markup to place in the panel.
 * @param {string} [variant] - Bootstrap contextual class suffix.
 */
function renderResult(html, variant) {
    const panel = document.getElementById("system_response");
    panel.className = "result-panel" + (variant ? " result-" + variant : "");
    panel.innerHTML = html;
}

/**
 * Turn the JSON error envelope into readable markup.
 *
 * @param {Object} payload - Body returned by server.py.
 * @returns {string} Markup describing the failure.
 */
function renderError(payload) {
    const error = (payload && payload.error) || {};
    const status = error.status || "";
    const message = error.message || "The request could not be completed.";
    return (
        "<p class='mb-1'><strong>Error " + status + "</strong></p>" +
        "<p class='mb-0'>" + message + "</p>"
    );
}

/**
 * Disable the submit button and show a pending state.
 */
function setBusy(busy) {
    const button = document.getElementById("detectButton");
    button.disabled = busy;
    button.textContent = busy ? "Analyzing..." : "Run Emotion Detection";
}

/**
 * Read a response body as text and parse it when it happens to be JSON.
 *
 * server.py answers a successful detection with a plain text sentence and
 * every failure with a JSON error envelope, so the body is read as text first
 * and only then handed to JSON.parse.
 *
 * @param {Response} response - Fetch response to consume.
 * @returns {Promise<{raw: string, payload: Object|null}>} Body in both forms.
 */
async function readBody(response) {
    const raw = await response.text();
    try {
        return { raw: raw, payload: JSON.parse(raw) };
    } catch (parseError) {
        return { raw: raw, payload: null };
    }
}

/**
 * Send the text to the emotion detector and render the outcome.
 *
 * @returns {Promise<void>} Resolves once the result panel has been updated.
 */
async function RunSentimentAnalysis() {
    const text = document.getElementById("textToAnalyze").value;

    if (!text.trim()) {
        renderResult(
            "<p class='mb-0'>Please enter some text before running the analysis.</p>",
            "error"
        );
        return;
    }

    setBusy(true);
    renderResult("<p class='mb-0'>Analyzing feedback...</p>");

    try {
        const response = await fetch(ENDPOINT + "?textToAnalyse=" + encodeURIComponent(text), {
            method: "GET",
            headers: { Accept: "application/json, text/plain" }
        });
        const body = await readBody(response);

        if (response.ok) {
            renderResult("<p class='mb-0'>" + body.raw + "</p>", "success");
        } else if (body.payload) {
            renderResult(renderError(body.payload), "error");
        } else {
            renderResult("<p class='mb-0'>Request failed with HTTP " + response.status + ".</p>", "error");
        }
    } catch (networkError) {
        renderResult(
            "<p class='mb-0'>Could not reach the emotion detector. Is the server running?</p>",
            "error"
        );
    } finally {
        setBusy(false);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("detectButton").addEventListener("click", RunSentimentAnalysis);
    document.getElementById("textToAnalyze").addEventListener("keydown", (event) => {
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            RunSentimentAnalysis();
        }
    });
});
