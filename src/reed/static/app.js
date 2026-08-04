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

/* ------------------------------------------------------------------ utils */

const apiKey = () => localStorage.getItem("reed.apiKey") || "";

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
    const health = await api("/health");
    const ok = health.status === "ok";
    els.statusDot.className = `dot ${ok ? "ok" : "bad"}`;
    els.statusText.textContent = `${health.profile} · ${health.chat_model}`;
    els.statusText.title = ok ? "" : health.vector_store;
  } catch (error) {
    els.statusDot.className = "dot bad";
    els.statusText.textContent = "unreachable";
    els.statusText.title = error.message;
  }
}

/* --------------------------------------------------------------- documents */

let pollTimer = null;

function renderDocuments(documents) {
  els.docList.replaceChildren();
  els.docEmpty.hidden = documents.length > 0;
  els.docCounter.textContent = String(documents.length);

  for (const doc of documents) {
    const item = el("li", "doc");
    item.dataset.docId = doc.id;
    item.append(el("span", "doc-name", doc.filename));

    const remove = el("button", "doc-remove", "×");
    remove.title = "Remove";
    remove.addEventListener("click", () => removeDocument(doc.id));
    item.append(remove);

    const busyDoc = doc.status === "pending" || doc.status === "processing";
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

  const stillWorking = documents.some((d) => d.status === "pending" || d.status === "processing");
  clearTimeout(pollTimer);
  if (stillWorking) pollTimer = setTimeout(refreshDocuments, 1200);
}

async function refreshDocuments() {
  try {
    const { documents } = await api("/v1/documents");
    renderDocuments(documents);
  } catch (error) {
    console.error("could not list documents", error);
  }
}

async function removeDocument(id) {
  try {
    await api(`/v1/documents/${id}`, { method: "DELETE" });
  } catch (error) {
    alert(`Could not delete: ${error.message}`);
  }
  refreshDocuments();
}

async function uploadFiles(files) {
  for (const file of files) {
    const body = new FormData();
    body.append("file", file);
    try {
      await api("/v1/documents", { method: "POST", body });
    } catch (error) {
      if (!/already been ingested/i.test(error.message)) {
        alert(`${file.name}: ${error.message}`);
      }
    }
  }
  refreshDocuments();
}

/* -------------------------------------------------------------------- chat */

function renderAnswer(container, text, sourceNodes) {
  // Turn [n] markers into buttons that reveal the source they point at.
  container.replaceChildren();
  const pattern = /\[(\d{1,2})\]/g;
  let cursor = 0;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      container.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    const number = Number(match[1]);
    const chip = el("button", "cite", String(number));
    chip.type = "button";
    chip.addEventListener("click", () => {
      const target = sourceNodes[number - 1];
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "nearest" });
      target.classList.add("flash");
      setTimeout(() => target.classList.remove("flash"), 1400);
    });
    container.append(chip);
    cursor = match.index + match[0].length;
  }
  container.append(document.createTextNode(text.slice(cursor)));
}

function renderSources(container, sources) {
  container.replaceChildren();
  if (!sources.length) return [];

  const label = sources.length === 1 ? "1 source" : `${sources.length} sources`;
  container.append(el("span", "sources-label", label));
  return sources.map((source) => {
    const card = el("div", "source");
    card.append(el("span", "source-n", `[${source.n}]`));

    const where = el("div", "source-where", source.filename + (source.page ? `, p. ${source.page}` : ""));
    where.append(el("span", "source-score", source.score.toFixed(3)));
    card.append(where);
    card.append(el("div", "source-snippet", source.snippet));

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
  els.send.disabled = true;

  addMessage("user").bubble.textContent = question;
  const { bubble, sources: sourcesBox } = addMessage("assistant");
  bubble.classList.add("typing");
  scrollToBottom();

  let answer = "";
  let sourceNodes = [];

  try {
    const response = await fetch("/v1/ask", {
      method: "POST",
      headers: headers({ "Content-Type": "application/json" }),
      body: JSON.stringify({ question, history: history.slice(-6), stream: true }),
    });
    if (!response.ok || !response.body) throw new Error(`${response.status} ${response.statusText}`);

    for await (const event of readEvents(response.body)) {
      if (event.name === "sources") {
        sourceNodes = renderSources(sourcesBox, event.data.sources);
      } else if (event.name === "token") {
        answer += event.data.t;
        renderAnswer(bubble, answer, sourceNodes);
        scrollToBottom();
      } else if (event.name === "done") {
        answer = event.data.answer || answer;
        renderAnswer(bubble, answer, sourceNodes);
      } else if (event.name === "error") {
        throw new Error(event.data.message);
      }
    }

    history.push({ role: "user", content: question }, { role: "assistant", content: answer });
  } catch (error) {
    bubble.classList.add("failed");
    bubble.textContent = `Something went wrong: ${error.message}`;
  } finally {
    bubble.classList.remove("typing");
    busy = false;
    els.send.disabled = false;
    scrollToBottom();
  }
}

/** Parse an SSE byte stream into {name, data} objects. */
async function* readEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
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
  if (value) localStorage.setItem("reed.apiKey", value);
  else localStorage.removeItem("reed.apiKey");
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
