"use strict";
const $ = (id) => document.getElementById(id);
const names = {shufflenetv2_x0_5: "ShuffleNet", mobilenetv2_x0_5: "MobileNet", resnet56: "ResNet56"};
let selected = null;
let previewURL = null;
let busy = false;
function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
function resetResult() {
  $("result").hidden = true;
  $("empty").hidden = false;
}
function choose(file) {
  if (busy) return;
  selected = null;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  $("preview").hidden = true;
  $("placeholder").hidden = false;
  $("filename").textContent = "선택한 이미지 없음";
  $("clear").hidden = true;
  $("run").disabled = true;
  resetResult();
  message("");
  if (!file) return;
  if (!["image/jpeg", "image/png"].includes(file.type)) {
    message("JPEG 또는 PNG 이미지를 선택하세요.", true); return;
  }
  if (file.size > 5 * 1024 * 1024) {
    message("이미지는 5MiB 이하여야 합니다.", true); return;
  }
  selected = file;
  previewURL = URL.createObjectURL(file);
  $("preview").src = previewURL;
  $("preview").hidden = false;
  $("placeholder").hidden = true;
  $("filename").textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KiB`;
  $("clear").hidden = false;
  $("run").disabled = false;
}
$("file").addEventListener("change", (event) => choose(event.target.files[0]));
$("clear").addEventListener("click", () => { choose(null); $("file").value = ""; });
$("preview").addEventListener("error", () => { choose(null); message("이미지를 읽을 수 없습니다.", true); });
for (const eventName of ["dragenter", "dragover", "dragleave", "drop"]) {
  $("dropzone").addEventListener(eventName, (event) => {
    event.preventDefault();
    $("dropzone").classList.toggle("drag", ["dragenter", "dragover"].includes(eventName));
    if (eventName === "drop") {
      if (event.dataTransfer.files.length !== 1) {
        message("이미지 한 장을 선택하세요.", true); return;
      }
      choose(event.dataTransfer.files[0]);
    }
  });
}
async function health() {
  try {
    const response = await fetch("/health/ready", {signal: AbortSignal.timeout(5000)});
    const ready = response.ok && (await response.json()).ready;
    $("health").textContent = ready ? "추론 서비스 준비 완료" : "추론 서비스 준비 중";
    $("health").className = `status ${ready ? "ready" : "error"}`;
  } catch {
    $("health").textContent = "서비스 연결 확인 필요";
    $("health").className = "status error";
  }
}
function render(result) {
  $("prediction").textContent = result.prediction.label;
  $("class-name").textContent = result.prediction.class_name;
  $("score").textContent = `${(result.prediction.score * 100).toFixed(1)}%`;
  $("top-k").replaceChildren(...result.top_k.map((item) => {
    const row = document.createElement("div"); row.className = "rank";
    const label = document.createElement("span"); label.textContent = item.label;
    const bar = document.createElement("div"); bar.className = "bar";
    const fill = document.createElement("i"); fill.style.width = `${item.score * 100}%`;
    bar.append(fill);
    const score = document.createElement("span"); score.textContent = `${(item.score * 100).toFixed(1)}%`;
    row.append(label, bar, score); return row;
  }));
  document.querySelectorAll("[data-expert]").forEach((element) => {
    const active = element.dataset.expert === result.expert_id;
    element.classList.toggle("selected", active);
    element.setAttribute("aria-label", `${names[element.dataset.expert]}${active ? " 선택됨" : ""}`);
  });
  $("route-note").textContent = result.extractor_reused
    ? "ShuffleNet 선택 · Router 내부에서 계산한 분류 결과를 재사용했습니다."
    : `${names[result.expert_id]} 선택 · binary 요청으로 외부 expert를 호출했습니다.`;
  for (const key of ["total", "extractor", "dispatcher", "expert"]) {
    $(`${key}-time`).textContent = `${result.timings[`${key}_ms`].toFixed(2)} ms`;
  }
  $("request-id").textContent = `REQUEST ${result.request_id}`;
  $("empty").hidden = true; $("result").hidden = false;
}
$("run").addEventListener("click", async () => {
  if (!selected || busy) return;
  busy = true; $("run").disabled = true; $("file").disabled = true; $("clear").disabled = true;
  $("dropzone").classList.add("busy"); $("result-title").setAttribute("aria-busy", "true");
  resetResult(); message("모델이 이미지를 분석하고 있습니다…");
  try {
    const form = new FormData(); form.append("file", selected);
    const response = await fetch("/api/predict", {
      method: "POST", body: form, signal: AbortSignal.timeout(35000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "분류에 실패했습니다.");
    render(result); message("분류를 완료했습니다.");
  } catch (error) {
    message(error.name === "TimeoutError" ? "응답 시간이 초과됐습니다. 다시 시도하세요."
      : error.message || "서비스 연결을 확인한 뒤 다시 시도하세요.", true);
  } finally {
    busy = false; $("run").disabled = !selected; $("file").disabled = false; $("clear").disabled = false;
    $("dropzone").classList.remove("busy"); $("result-title").removeAttribute("aria-busy");
    health();
  }
});
health();
setInterval(health, 15000);
