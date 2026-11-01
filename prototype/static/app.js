// All text is written with textContent, so model output can't run as code.
const $ = (id) => document.getElementById(id);
const TEXT_LABELS = { ask: "Sorunuz", diagnose: "Kurgusal durumu yazın" };
let mode = "ask";

function setMode(next) {
  mode = next;
  for (const tab of document.querySelectorAll("[role=tab]")) {
    const selected = tab.dataset.mode === mode;
    tab.setAttribute("aria-selected", String(selected));
    if (selected) $("panel").setAttribute("aria-labelledby", tab.id);
  }
  $("text-fields").hidden = mode === "dose";
  $("dose-fields").hidden = mode !== "dose";
  if (mode !== "dose") $("text-label").textContent = TEXT_LABELS[mode];
  $("result").hidden = true;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showSources(sources) {
  $("sources").replaceChildren(...sources.map((source) => {
    const block = element("article", "", "source");
    block.append(element("h2", `Kaynak: ${source.title}`));
    block.append(element("p", `Test taslağı, uzman onaylı değil. Tarih: ${source.pack_date}`, "meta"));
    const list = document.createElement("ol");
    list.append(...Object.values(source.sentences).map((sentence) => element("li", sentence)));
    block.append(list);
    return block;
  }));
}

function show(result) {
  const labels = [];
  if (result.kind === "answer" || result.kind === "possible_diagnosis") labels.push("Yapay zekâ yanıtı, yanlış olabilir");
  if (result.kind === "source_only") labels.push("Yalnızca kaynak");
  if (result.low_match) labels.push("Zayıf eşleşme");
  $("labels").replaceChildren(...labels.map((text) => element("span", text, "label")));
  $("body").textContent = result.body;
  $("warnings").replaceChildren(...(result.warnings || []).map((text) => element("li", text)));
  showSources(result.sources || []);
  $("result").hidden = false;
}

async function send(values) {
  const response = await fetch("/api/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(values),
  });
  if (!response.ok) throw new Error(String(response.status));
  return response.json();
}

$("panel").addEventListener("submit", async (event) => {
  event.preventDefault();
  const values = mode === "dose"
    ? { mode, medicine: $("medicine").value, age: $("age").value, weight_kg: $("weight").value }
    : { mode, text: $("text").value };
  $("submit").disabled = true;
  $("submit").textContent = "Yanıt hazırlanıyor…";
  try {
    show(await send(values));
  } catch {
    show({ kind: "error", body: "İstek tamamlanamadı. Sunucunun çalıştığını kontrol edin.", sources: [], warnings: [] });
  } finally {
    $("submit").disabled = false;
    $("submit").textContent = "Gönder";
  }
});

for (const tab of document.querySelectorAll("[role=tab]")) {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
}

fetch("/api/medicines").then((response) => response.json()).then((medicines) => {
  $("medicine").replaceChildren(...medicines.map((medicine) => {
    const option = element("option", medicine.name);
    option.value = medicine.id;
    return option;
  }));
}).catch(() => {});
