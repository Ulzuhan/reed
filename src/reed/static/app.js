/* Reed chat UI — no framework, no build step.
 *
 * Streaming uses fetch + ReadableStream rather than EventSource, because
 * EventSource can only issue GETs and cannot send the X-API-Key header.
 */

const $ = (id) => document.getElementById(id);

const els = {
  messages: $("messages"),
  welcome: $("welcome"),
  samples: $("samples"),
  composer: $("composer"),
  question: $("question"),
  send: $("send"),
  fileInput: $("file-input"),
  uploadButton: $("upload-button"),
  uploader: $("uploader"),
  uploadProgress: $("upload-progress"),
  docList: $("doc-list"),
  docEmpty: $("doc-empty"),
  docCounter: $("doc-counter"),
  statusDot: $("status-dot"),
  statusText: $("status-text"),
  settingsButton: $("settings-button"),
  settingsDialog: $("settings-dialog"),
  apiKeyInput: $("api-key-input"),
  saveKey: $("save-key"),
};

const SAMPLE_QUESTIONS = [
  "What does this document say about deadlines?",
  "Summarise the key policies.",
  "Who is responsible for approvals?",
];

const history = [];
let busy = false;
let askController = null;

// Migrate once from the older persistent storage. A tab-scoped credential
// limits exposure to future same-origin scripts and disappears when the tab closes.
const legacyApiKey = localStorage.getItem("reed.apiKey");
if (legacyApiKey && !sessionStorage.getItem("reed.apiKey")) {
  sessionStorage.setItem("reed.apiKey", legacyApiKey);
}
localStorage.removeItem("reed.apiKey");

/* ------------------------------------------------------------------ utils */

const apiKey = () => sessionStorage.getItem("reed.apiKey") || "";

function headers(extra = {}) {
  const key = apiKey();
  return key ? { ...extra, "X-API-Key": key } : extra;
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: headers(options.headers) });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : body.detail.message;
    } catch {
      /* keep the status line */
    }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const scrollToBottom = () => {
  els.messages.scrollTop = els.messages.scrollHeight;
};

/* ------------------------------------------------------------------ health */

async function refreshStatus() {
  try {
    const response = await fetch("/ready", { headers: headers() });
    const health = await response.json();
    const ok = response.ok && health.status === "ok";
    els.statusDot.className = `dot ${ok ? "ok" : "bad"}`;
    els.statusText.textContent = health.profile
      ? `${health.profile} · ${health.chat_model}`
      : ok
        ? "ready · details protected"
        : "degraded · details protected";
    els.statusText.title = ok ? "" : health.vector_store;
  } catch (error) {
    els.statusDot.className = "dot bad";
    els.statusText.textContent = "unreachable";
    els.statusText.title = error.message;
  }
}

/* --------------------------------------------------------------- documents */

let pollTimer = null;

function renderDocuments(documents, total = documents.length) {
  els.docList.replaceChildren();
  els.docEmpty.hidden = documents.length > 0;
  els.docCounter.textContent = documents.length === total ? String(total) : `${documents.length}/${total}`;

  for (const doc of documents) {
    const item = el("li", "doc");
    item.dataset.docId = doc.id;
    item.append(el("span", "doc-name", doc.filename));

    const remove = el("button", "doc-remove", "×");
    remove.title = "Remove";
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${doc.filename}`);
    remove.addEventListener("click", () => removeDocument(doc.id));
    item.append(remove);

    const busyDoc = ["pending", "processing", "queued", "parsing", "embedding", "indexing", "deleting"].includes(doc.status);
    const meta = el("span", `doc-meta${doc.status === "error" ? " error" : ""}`);
    if (busyDoc) {
      meta.append(el("span", "spin", "⟳"), document.createTextNode(` ${doc.status}…`));
    } else if (doc.status === "error") {
      meta.textContent = doc.error || "failed";
    } else {
      const pages = doc.pages ? ` · ${doc.pages} pages` : "";
      meta.textContent = `${doc.chunks} chunks${pages}`;
    }
    item.append(meta);
    els.docList.append(item);
  }

  const stillWorking = documents.some((d) =>
    ["pending", "processing", "queued", "parsing", "embedding", "indexing", "deleting"].includes(d.status),
  );
  clearTimeout(pollTimer);
  if (stillWorking) pollTimer = setTimeout(refreshDocuments, 1200);
}

async function refreshDocuments() {
  try {
    const documents = [];
    const limit = 100;
    let total = 0;
    do {
      const page = await api(`/v1/documents?limit=${limit}&offset=${documents.length}`);
      documents.push(...page.documents);
      total = page.total;
      if (!page.documents.length) break;
    } while (documents.length < total);
    renderDocuments(documents, total);
  } catch (error) {
    console.error("could not list documents", error);
  }
}

async function removeDocument(id) {
  if (!window.confirm("Delete this document and all of its indexed chunks?")) return;
  try {
    await api(`/v1/documents/${id}`, { method: "DELETE" });
  } catch (error) {
    alert(`Could not delete: ${error.message}`);
  }
  refreshDocuments();
}

async function uploadFiles(files) {
  for (const file of files) {
    try {
      await uploadFile(file);
    } catch (error) {
      if (!/already been ingested/i.test(error.message)) {
        alert(`${file.name}: ${error.message}`);
      }
    }
  }
  els.uploadProgress.textContent = "or drop PDF, DOCX, HTML, Markdown and text files here";
  refreshDocuments();
}

function uploadFile(file) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/v1/documents");
    const key = apiKey();
    if (key) request.setRequestHeader("X-API-Key", key);
    request.upload.addEventListener("progress", (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.round((event.loaded / event.total) * 100);
      els.uploadProgress.textContent = `${file.name}: uploading ${percent}%`;
    });
    request.addEventListener("load", () => {
      let body = null;
      try {
        body = JSON.parse(request.responseText);
      } catch {
        /* retain the HTTP status as fallback */
      }
      if (request.status >= 200 && request.status < 300) {
        els.uploadProgress.textContent = `${file.name}: queued for ingestion`;
        resolve(body);
        return;
      }
      const detail = body?.detail;
      reject(
        new Error(
          typeof detail === "string"
            ? detail
            : detail?.message || `${request.status} ${request.statusText}`,
        ),
      );
    });
    request.addEventListener("error", () => reject(new Error("Upload connection failed")));
    const form = new FormData();
    form.append("file", file);
    request.send(form);
  });
}

/* -------------------------------------------------------------------- chat */

// Citation markers become buttons; **bold** becomes bold. Everything else is
// inserted as text, so a model that emits markup cannot inject it.
const INLINE = /\[(\d{1,2})\]|\*\*([^*]+)\*\*/g;

function renderAnswer(container, text, sourceNodes) {
  container.replaceChildren();
  let cursor = 0;
  let match;

  while ((match = INLINE.exec(text)) !== null) {
    if (match.index > cursor) {
      container.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    container.append(
      match[1] !== undefined ? citation(Number(match[1]), sourceNodes) : el("strong", null, match[2]),
    );
    cursor = match.index + match[0].length;
  }
  INLINE.lastIndex = 0;
  container.append(document.createTextNode(text.slice(cursor)));
}

function citation(number, sourceNodes) {
  const chip = el("button", "cite", String(number));
  chip.type = "button";
  chip.title = `Jump to source ${number}`;
  if (!sourceNodes[number - 1]) {
    chip.classList.add("invalid");
    chip.title = `Unavailable source ${number}`;
  }
  chip.addEventListener("click", () => {
    const target = sourceNodes[number - 1];
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    target.classList.add("flash");
    setTimeout(() => target.classList.remove("flash"), 1400);
  });
  return chip;
}

function renderSources(container, sources) {
  container.replaceChildren();
  if (!sources.length) return [];

  const label = sources.length === 1 ? "1 source" : `${sources.length} sources`;
  container.append(el("span", "sources-label", label));
  return sources.map((source) => {
    const card = el("div", "source");
    card.append(el("span", "source-n", `[${source.n}]`));

    // The same label the model saw, so a citation is traceable to its passage.
    const where = el("div", "source-where", source.location || source.filename);
    where.append(el("span", "source-score", source.score.toFixed(3)));
    card.append(where);
    card.append(el("div", "source-snippet", source.snippet));

    const details = el("details", "source-details");
    details.append(el("summary", null, "Show complete excerpt"));
    details.append(el("pre", "source-excerpt", source.excerpt || source.snippet));
    card.append(details);

    container.append(card);
    highlightDocument(source.doc_id);
    return card;
  });
}

function highlightDocument(docId) {
  const node = els.docList.querySelector(`[data-doc-id="${docId}"]`);
  if (!node) return;
  node.classList.add("highlight");
  setTimeout(() => node.classList.remove("highlight"), 2000);
}

function addMessage(role) {
  els.welcome?.remove();
  const message = el("div", `message ${role}`);
  const bubble = el("div", "bubble");
  message.append(bubble);
  const sources = el("div", "sources");
  message.append(sources);
  els.messages.append(message);
  return { bubble, sources };
}

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  askController = new AbortController();
  els.send.textContent = "Stop";
  els.send.title = "Stop generation";

  addMessage("user").bubble.textContent = question;
  const { bubble, sources: sourcesBox } = addMessage("assistant");
  bubble.classList.add("typing");
  scrollToBottom();

  let answer = "";
  let sourceNodes = [];
  let completed = false;

  try {
    const response = await fetch("/v1/ask", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question, history: history.slice(-6), stream: true }),
      signal: askController.signal,
    });
    if (!response.ok || !response.body) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        const detail = body.detail;
        message = typeof detail === "string" ? detail : detail?.message || message;
      } catch {
        /* retain status */
      }
      throw new Error(message);
    }

    for await (const event of readEvents(response.body)) {
      if (event.name === "sources") {
        sourceNodes = renderSources(sourcesBox, event.data.sources);
      } else if (event.name === "token") {
        answer += event.data.t;
        renderAnswer(bubble, answer, sourceNodes);
        scrollToBottom();
      } else if (event.name === "done") {
        completed = true;
        answer = event.data.answer || answer;
        renderAnswer(bubble, answer, sourceNodes);
        renderCitationStatus(sourcesBox, event.data.citation_status, event.data.citation_warnings);
      } else if (event.name === "error") {
        throw new Error(event.data.message);
      }
    }

    if (!completed) throw new Error("The response stream ended before completion");

    history.push({ role: "user", content: question }, { role: "assistant", content: answer });
  } catch (error) {
    if (error.name === "AbortError") {
      bubble.textContent = answer ? `${answer}\n\nStopped.` : "Stopped.";
    } else {
      bubble.classList.add("failed");
      bubble.textContent = `Something went wrong: ${error.message}`;
    }
  } finally {
    bubble.classList.remove("typing");
    busy = false;
    askController = null;
    els.send.textContent = "Ask";
    els.send.title = "Send (Enter)";
    scrollToBottom();
  }
}

function renderCitationStatus(container, status, warnings = []) {
  if (!status || status === "valid" || status === "not_applicable") return;
  const message =
    warnings.join(" ") ||
    (status === "missing" ? "The answer contains no citations." : "Some citations are invalid.");
  container.prepend(el("div", `citation-warning ${status}`, message));
}

/** Parse an SSE byte stream into {name, data} objects. */
async function* readEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      buffer += decoder.decode();
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      let name = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith(":")) continue; // heartbeat
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) yield { name, data: JSON.parse(dataLines.join("\n")) };
    }
  }
}

/* ------------------------------------------------------------------- wiring */

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy) {
    askController?.abort();
    return;
  }
  const question = els.question.value;
  els.question.value = "";
  els.question.style.height = "auto";
  ask(question);
});

els.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

els.question.addEventListener("input", () => {
  els.question.style.height = "auto";
  els.question.style.height = `${els.question.scrollHeight}px`;
});

els.uploadButton.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", () => {
  uploadFiles([...els.fileInput.files]);
  els.fileInput.value = "";
});

for (const [event, active] of [
  ["dragenter", true],
  ["dragover", true],
  ["dragleave", false],
  ["drop", false],
]) {
  els.uploader.addEventListener(event, (e) => {
    e.preventDefault();
    els.uploader.classList.toggle("dragging", active);
    if (event === "drop") uploadFiles([...e.dataTransfer.files]);
  });
}

els.settingsButton.addEventListener("click", () => {
  els.apiKeyInput.value = apiKey();
  els.settingsDialog.showModal();
});

els.saveKey.addEventListener("click", () => {
  const value = els.apiKeyInput.value.trim();
  if (value) sessionStorage.setItem("reed.apiKey", value);
  else sessionStorage.removeItem("reed.apiKey");
  refreshStatus();
  refreshDocuments();
});

for (const question of SAMPLE_QUESTIONS) {
  const chip = el("button", "sample", question);
  chip.type = "button";
  chip.addEventListener("click", () => ask(question));
  els.samples.append(chip);
}

refreshStatus();
refreshDocuments();
setInterval(refreshStatus, 30000);
