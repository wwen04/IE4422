const accuracyMessage = document.querySelector("#accuracy-message");
const accuracyContent = document.querySelector("#accuracy-content");
const hintsMessage = document.querySelector("#hints-message");
const hintsContent = document.querySelector("#hints-content");
const ratingsMessage = document.querySelector("#ratings-message");
const ratingsContent = document.querySelector("#ratings-content");
const accuracyItemSelect = document.querySelector("#accuracy-item-select");
const hintItemSelect = document.querySelector("#hint-item-select");
const dHintsMessage = document.querySelector("#d-hints-message");
const dHintsContent = document.querySelector("#d-hints-content");
const dHintCaseSelect = document.querySelector("#d-hint-case-select");
const runEvaluationButton = document.querySelector("#run-evaluation-button");
const evaluationLimitInput = document.querySelector("#evaluation-limit");
const evaluationStatus = document.querySelector("#evaluation-status");
const evaluationError = document.querySelector("#evaluation-error");

let accuracyItems = [];
let hintItems = [];
let dHintCases = [];
let evaluationBusy = false;

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function setMessage(element, title, detail, command) {
  element.replaceChildren();
  const heading = document.createElement("strong");
  heading.textContent = title;
  const description = document.createElement("span");
  description.textContent = detail;
  element.append(heading, description);
  if (command) {
    const code = document.createElement("code");
    code.textContent = command;
    element.append(code);
  }
  element.hidden = false;
}

function updateEvaluationStatus(job) {
  const percent = Math.max(0, Math.min(100, Number(job.progress) || 0));
  document.querySelector("#evaluation-stage").textContent =
    job.stage_label || "Preparing comparison…";
  document.querySelector("#evaluation-progress-label").textContent = `${percent}%`;
  document.querySelector("#evaluation-progress-bar").style.width = `${percent}%`;
  const progressBar = document.querySelector(".evaluation-progress-track");
  progressBar.setAttribute("aria-valuenow", String(percent));
  document.querySelector("#evaluation-detail").textContent = job.current_item
    ? `Question ${job.completed} of ${job.total} · ${job.current_item}`
    : job.result
      ? `${job.result.accuracy_questions} accuracy results · ` +
        `${job.result.hint_questions} paired hint sets · ` +
        (job.result.hints_generated
          ? "new hint set created"
          : "existing hint set kept, including any saved ratings")
      : "";
}

async function pollEvaluation(jobId) {
  while (true) {
    const job = await requestJson(`/api/evaluations/${encodeURIComponent(jobId)}`);
    updateEvaluationStatus(job);
    if (job.status === "complete") {
      await Promise.all([loadAccuracy(), loadHints()]);
      return;
    }
    if (job.status === "failed") {
      throw new Error(job.error || "The comparison run failed.");
    }
    await new Promise(resolve => setTimeout(resolve, 1200));
  }
}

async function runEvaluation() {
  if (evaluationBusy) return;
  const limit = Number(evaluationLimitInput.value);
  if (!Number.isInteger(limit) || limit < 1 || limit > 25) {
    evaluationError.textContent = "Choose a whole number from 1 to 25.";
    evaluationError.hidden = false;
    return;
  }
  evaluationBusy = true;
  runEvaluationButton.disabled = true;
  evaluationLimitInput.disabled = true;
  evaluationError.hidden = true;
  evaluationStatus.hidden = false;
  document.querySelector("#evaluation-detail").textContent =
    "Uses the same Qwen model. The first run may need time to load it.";
  updateEvaluationStatus({
    stage_label: "Starting validation comparison…",
    progress: 0,
  });
  document.querySelector("#evaluation-detail").textContent =
    "Uses the same Qwen model. The first run may need time to load it.";
  try {
    const started = await requestJson("/api/evaluations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit }),
    });
    await pollEvaluation(started.job_id);
  } catch (error) {
    evaluationError.textContent = error.message;
    evaluationError.hidden = false;
    document.querySelector("#evaluation-stage").textContent = "Evaluation stopped";
  } finally {
    evaluationBusy = false;
    runEvaluationButton.disabled = false;
    evaluationLimitInput.disabled = false;
  }
}

function percent(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0 || number > 100) {
    throw new Error("The accuracy report contains an invalid percentage.");
  }
  return `${number.toFixed(2)}%`;
}

function renderAccuracy(report) {
  if (!report.available) return;
  if (!Array.isArray(report.items) || !report.items.length
      || !report.baseline || !report.evidence_guided || !report.paired) {
    throw new Error("The accuracy report is missing paired results.");
  }
  accuracyItems = report.items;
  const total = Number(report.n);
  if (!Number.isInteger(total) || total !== accuracyItems.length) {
    throw new Error("The accuracy report has an inconsistent question count.");
  }
  const direct = report.baseline;
  const guided = report.evidence_guided;
  if (!Number.isInteger(direct.correct) || !Number.isInteger(guided.correct)
      || direct.correct !== accuracyItems.filter(item =>
        item.baseline_correct === true).length
      || guided.correct !== accuracyItems.filter(item =>
        item.evidence_guided_correct === true).length) {
    throw new Error("Accuracy totals do not match the paired answer results.");
  }
  const expected = {
    both_correct: accuracyItems.filter(item =>
      item.baseline_correct && item.evidence_guided_correct).length,
    guided_only: accuracyItems.filter(item =>
      !item.baseline_correct && item.evidence_guided_correct).length,
    direct_only: accuracyItems.filter(item =>
      item.baseline_correct && !item.evidence_guided_correct).length,
    both_incorrect: accuracyItems.filter(item =>
      !item.baseline_correct && !item.evidence_guided_correct).length,
  };
  if (direct.correct !== expected.both_correct + expected.direct_only
      || guided.correct !== expected.both_correct + expected.guided_only
      || Object.entries(expected).some(([key, count]) =>
        report.paired[key === "guided_only" ? "evidence_guided_only_correct"
          : key === "direct_only" ? "baseline_only_correct" : key] !== count)) {
    throw new Error("Paired outcomes do not match the answer results.");
  }
  const directPercent = direct.correct / total * 100;
  const guidedPercent = guided.correct / total * 100;
  const delta = Number(report.delta_percentage_points);
  if (Math.abs(Number(direct.accuracy_percent) - directPercent) > 0.01
      || Math.abs(Number(guided.accuracy_percent) - guidedPercent) > 0.01
      || !Number.isFinite(delta)
      || Math.abs(delta - (guidedPercent - directPercent)) > 0.01) {
    throw new Error("Accuracy percentages do not match the paired answer results.");
  }
  accuracyMessage.hidden = true;
  accuracyContent.hidden = false;
  document.querySelector("#accuracy-meta").textContent =
    `${report.split} · ${total} questions · ${report.model || "model not recorded"} · ${report.created_at || ""}`;
  for (const [prefix, metric, bar] of [
    ["direct", direct, "direct-bar"],
    ["guided", guided, "guided-bar"],
  ]) {
    const value = Number(metric.accuracy_percent);
    document.querySelector(`#${prefix}-accuracy`).textContent = percent(value);
    document.querySelector(`#${prefix}-count`).textContent =
      `${metric.correct} of ${total} answers correct`;
    document.querySelector(`#${bar}`).style.width = `${value}%`;
  }
  const deltaCard = document.querySelector("#delta-card");
  deltaCard.classList.toggle("positive", delta > 0);
  deltaCard.classList.toggle("negative", delta < 0);
  document.querySelector("#accuracy-delta").textContent =
    `${delta > 0 ? "+" : ""}${delta.toFixed(2)} pp`;
  document.querySelector("#delta-caption").textContent =
    `${expected.guided_only} improved · ${expected.direct_only} regressed · ${expected.both_correct} stayed correct`;

  const outcomeLabels = [
    ["both_correct", "Both correct", "#4a805e"],
    ["guided_only", "Only evidence-guided correct", "#a5c9a9"],
    ["direct_only", "Only direct correct", "#d99a68"],
    ["both_incorrect", "Both incorrect", "#d8dfd7"],
  ];
  const bar = document.querySelector("#paired-bar");
  const legend = document.querySelector("#paired-legend");
  bar.replaceChildren();
  legend.replaceChildren();
  for (const [key, label, color] of outcomeLabels) {
    const count = expected[key];
    const segment = document.createElement("span");
    segment.className = key.replaceAll("_", "-");
    segment.style.width = `${count / total * 100}%`;
    segment.title = `${label}: ${count}`;
    segment.setAttribute("aria-label", `${label}: ${count}`);
    bar.append(segment);
    const legendItem = document.createElement("div");
    legendItem.className = "paired-legend-item";
    const swatch = document.createElement("span");
    swatch.className = "paired-legend-swatch";
    swatch.style.backgroundColor = color;
    const text = document.createElement("span");
    text.append(document.createTextNode(`${label} `));
    const amount = document.createElement("strong");
    amount.textContent = String(count);
    text.append(amount);
    legendItem.append(swatch, text);
    legend.append(legendItem);
  }

  const splitRows = document.querySelector("#split-rows");
  splitRows.replaceChildren();
  for (const [split, metrics] of Object.entries(report.by_split || {})) {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = `${split} · ${metrics.n}`;
    const directCell = document.createElement("td");
    directCell.textContent = percent(metrics.baseline.accuracy_percent);
    const guidedCell = document.createElement("td");
    guidedCell.textContent = percent(metrics.evidence_guided.accuracy_percent);
    const deltaCell = document.createElement("td");
    const splitDelta = Number(metrics.delta_percentage_points);
    deltaCell.textContent = `${splitDelta > 0 ? "+" : ""}${splitDelta.toFixed(2)} pp`;
    row.append(label, directCell, guidedCell, deltaCell);
    splitRows.append(row);
  }
  accuracyItemSelect.replaceChildren();
  accuracyItems.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${item.split} · ${item.item_id}`;
    accuracyItemSelect.append(option);
  });
  renderAccuracyItem();
}

function outputCard(title, prediction, correct, response, evidence) {
  const card = document.createElement("article");
  card.className = "answer-output-card";
  const heading = document.createElement("h4");
  heading.textContent = title;
  const status = document.createElement("p");
  status.className = correct ? "correct-tag" : "incorrect-tag";
  status.textContent = `${correct ? "Correct" : "Incorrect"}${prediction ? ` · ${prediction}` : ""}`;
  const output = document.createElement("p");
  output.textContent = response || "(No model response)";
  card.append(heading, status, output);
  if (evidence) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Evidence used";
    const content = document.createElement("p");
    content.textContent = evidence;
    details.append(summary, content);
    card.append(details);
  }
  return card;
}

function renderAccuracyItem() {
  if (!accuracyItems.length) return;
  const item = accuracyItems[Number(accuracyItemSelect.value) || 0];
  const output = document.querySelector("#accuracy-item-output");
  output.replaceChildren(
    outputCard("Direct answer", item.baseline_prediction, item.baseline_correct,
      item.baseline_response),
    outputCard("Evidence-guided answer", item.evidence_guided_prediction,
      item.evidence_guided_correct, item.evidence_guided_response, item.visual_evidence),
  );
}

function ratingChips(ratings, level) {
  const dimensions = [
    ["factual_accuracy_1_to_4", "Accuracy"],
    ["image_groundedness_1_to_4", "Grounding"],
    ["pedagogical_usefulness_1_to_4", "Usefulness"],
  ];
  const chips = document.createElement("div");
  chips.className = "rating-inline";
  for (const [dimension, label] of dimensions) {
    const value = ratings[`level_${level}_${dimension}`];
    if (!value) continue;
    const chip = document.createElement("span");
    chip.className = "rating-chip";
    chip.append(document.createTextNode(`${label} `));
    const score = document.createElement("strong");
    score.textContent = `${value}/4`;
    chip.append(score);
    chips.append(chip);
  }
  const leakage = ratings[`level_${level}_answer_leakage_yes_no_unclear`];
  const unsupported = ratings[`level_${level}_unsupported_claim_yes_no`];
  for (const [label, value] of [["Leakage", leakage], ["Unsupported", unsupported]]) {
    if (!value) continue;
    const chip = document.createElement("span");
    chip.className = "rating-chip";
    chip.textContent = `${label}: ${value}`;
    chips.append(chip);
  }
  return chips;
}

function renderHintPair(item, index) {
  document.querySelector("#hint-item-position").textContent =
    ` · ${index + 1} of ${hintItems.length}`;
  document.querySelector("#hint-question").textContent = item.question;
  const context = document.querySelector("#hint-context");
  context.textContent = item.context || "";
  context.hidden = !item.context;
  const choices = document.querySelector("#hint-choices");
  choices.replaceChildren();
  for (const line of (item.choices || "").split("\n").filter(Boolean)) {
    const choice = document.createElement("span");
    choice.className = "hint-choice";
    const match = line.match(/^([A-Z0-9]+)\.\s*(.*)$/);
    if (match) {
      const id = document.createElement("strong");
      id.textContent = `${match[1]}.`;
      choice.append(id, document.createTextNode(match[2]));
    } else {
      choice.textContent = line;
    }
    choices.append(choice);
  }
  const image = document.querySelector("#hint-image");
  image.src = item.image_url;
  document.querySelector("#hint-image-link").href = item.image_url;
  document.querySelector("#hint-evidence").textContent =
    item.visual_evidence || "No visual-evidence transcript was saved.";
  const grid = document.querySelector("#hint-pair-grid");
  grid.replaceChildren();
  for (let level = 1; level <= 3; level += 1) {
    const row = document.createElement("article");
    row.className = "hint-level-row";
    const marker = document.createElement("div");
    marker.className = "level-marker";
    const number = document.createElement("span");
    number.textContent = `LEVEL 0${level}`;
    const name = document.createElement("strong");
    name.textContent = ["Notice", "Connect", "Reason"][level - 1];
    marker.append(number, name);
    row.append(marker);
    for (const [method, title, className] of [
      ["direct_visual", "No evidence", "hint-output-card"],
      ["evidence_guided", "With evidence", "hint-output-card guided-output"],
    ]) {
      const condition = item.conditions[method];
      const card = document.createElement("div");
      card.className = className;
      const header = document.createElement("div");
      header.className = "hint-output-heading";
      const label = document.createElement("strong");
      label.textContent = title;
      const scoreLabel = document.createElement("span");
      scoreLabel.textContent = "Generated hint";
      header.append(label, scoreLabel);
      const hint = document.createElement("p");
      hint.className = "hint-copy";
      hint.textContent = condition.hints[level - 1] || "(No hint saved)";
      card.append(header, hint, ratingChips(condition.ratings, level));
      row.append(card);
    }
    grid.append(row);
  }
}

function renderRatings(summary, ratingCounts) {
  const ratingsRows = document.querySelector("#ratings-rows");
  ratingsRows.replaceChildren();
  const labels = [
    ["factual_accuracy_1_to_4", "Accuracy"],
    ["image_groundedness_1_to_4", "Grounding"],
    ["pedagogical_usefulness_1_to_4", "Usefulness"],
  ];
  for (let level = 1; level <= 3; level += 1) {
    for (const [method, title, rowClass] of [
      ["direct_visual", "No evidence", ""],
      ["evidence_guided", "With evidence", "guided-rating-row"],
    ]) {
      const metrics = summary.methods[method];
      const levelSummary = metrics.levels[String(level)];
      const tr = document.createElement("tr");
      if (rowClass) tr.className = rowClass;
      if (method === "direct_visual") {
        const levelCell = document.createElement("th");
        levelCell.scope = "row";
        levelCell.rowSpan = 2;
        levelCell.textContent = `Level ${level}`;
        tr.append(levelCell);
      }
      const methodCell = document.createElement("td");
      methodCell.textContent = `${title} · ${metrics.n} hints`;
      tr.append(methodCell);
      for (const [dimension] of labels) {
        const cell = document.createElement("td");
        const value = levelSummary[dimension];
        if (value === null || value === undefined) {
          cell.className = "rating-blank";
          cell.textContent = "Not rated";
        } else {
          const rating = document.createElement("div");
          rating.className = "rating-average";
          const score = document.createElement("strong");
          score.textContent = Number(value).toFixed(2);
          const count = document.createElement("span");
          count.textContent = `n=${ratingCounts[method][String(level)][dimension]}`;
          rating.append(score, count);
          cell.append(rating);
        }
        tr.append(cell);
      }
      const leakageCell = document.createElement("td");
      const leakage = levelSummary.answer_leakage;
      leakageCell.textContent = leakage.rated_n
        ? `${leakage.yes} yes / ${leakage.rated_n} rated${leakage.unclear ? ` · ${leakage.unclear} unclear` : ""}`
        : "Not rated";
      const unsupportedCell = document.createElement("td");
      const unsupported = levelSummary.unsupported_claims;
      unsupportedCell.textContent = unsupported.rated_n
        ? `${unsupported.yes} yes / ${unsupported.rated_n} rated`
        : "Not rated";
      tr.append(leakageCell, unsupportedCell);
      ratingsRows.append(tr);
    }
  }
}

function renderHints(report) {
  if (!report.available) return;
  if (!Array.isArray(report.items) || !report.items.length || !report.summary) {
    throw new Error("The hint comparison set has no paired items.");
  }
  hintItems = report.items;
  hintsMessage.hidden = true;
  hintsContent.hidden = false;
  document.querySelector("#hints-meta").textContent =
    `${report.n_questions} paired validation questions · ${report.model || "model not recorded"} · ${report.created_at || ""}`;
  hintItemSelect.replaceChildren();
  hintItems.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index + 1}. ${item.item_id} · ${item.question}`;
    hintItemSelect.append(option);
  });
  renderHintPair(hintItems[0], 0);
  const rated = Object.values(report.rating_counts || {}).some(method =>
    Object.values(method).some(level => Object.values(level).some(count => count > 0)));
  document.querySelector("#ratings-summary-meta").textContent =
    `${report.summary.n_questions} questions in the paired review set · ` +
    `${report.summary.methods.direct_visual.n} direct and ` +
    `${report.summary.methods.evidence_guided.n} evidence-guided hint sets`;
  renderRatings(report.summary, report.rating_counts);
  ratingsContent.hidden = !rated;
  if (!rated) {
    setMessage(
      ratingsMessage,
      "No reviewer scores recorded yet.",
      "The paired hint outputs are ready to compare above. Complete ratings in the blinded review studio; this view summarizes them here once saved.",
    );
    const link = document.createElement("a");
    link.href = "/review";
    link.textContent = "Open the blinded hint review studio ↗";
    ratingsMessage.append(link);
  } else {
    ratingsMessage.hidden = true;
  }
}

function renderExperimentDCase(item, index) {
  document.querySelector("#d-hint-case-position").textContent =
    ` · ${index + 1} of ${dHintCases.length}`;
  document.querySelector("#d-hint-question").textContent = item.question;
  const context = document.querySelector("#d-hint-context");
  context.textContent = item.context || "";
  context.hidden = !item.context;

  const choices = document.querySelector("#d-hint-choices");
  choices.replaceChildren();
  for (const choice of item.choices) {
    const line = document.createElement("span");
    line.className = "hint-choice";
    const label = document.createElement("strong");
    label.textContent = `${choice.id}.`;
    line.append(label, document.createTextNode(choice.text));
    choices.append(line);
  }

  const image = document.querySelector("#d-hint-image");
  image.src = item.image_url;
  document.querySelector("#d-hint-image-link").href = item.image_url;
  document.querySelector("#d-hint-tier").textContent = String(item.tier);
  document.querySelector("#d-hint-record-id").textContent = item.record_id;
  document.querySelector("#d-hint-text").textContent = item.hint_text || "(Empty hint)";
}

function renderExperimentD(report) {
  if (!report.available) return;
  if (!Array.isArray(report.items) || !report.items.length
      || report.n_cases !== report.items.length
      || report.items.some(item =>
        !/^r_\d{6}$/.test(item.record_id)
        || ![1, 2, 3].includes(item.tier)
        || !Array.isArray(item.choices)
        || typeof item.hint_text !== "string"
        || typeof item.image_url !== "string")) {
    throw new Error("The Experiment D judging package is invalid or incomplete.");
  }
  dHintCases = report.items;
  dHintsMessage.hidden = true;
  dHintsContent.hidden = false;
  document.querySelector("#d-hints-meta").textContent =
    `${report.n_cases} Experiment D judging cases · ${report.name}`;
  dHintCaseSelect.replaceChildren();
  dHintCases.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index + 1}. ${item.record_id} · Tier ${item.tier}`;
    dHintCaseSelect.append(option);
  });
  renderExperimentDCase(dHintCases[0], 0);
}

async function loadAccuracy() {
  try {
    const report = await requestJson("/api/comparison");
    if (!report.available) return;
    renderAccuracy(report);
  } catch (error) {
    setMessage(accuracyMessage, "Accuracy results could not be loaded.", error.message);
  }
}

async function loadHints() {
  try {
    const report = await requestJson("/api/hint-comparison");
    if (!report.available) return;
    renderHints(report);
  } catch (error) {
    setMessage(hintsMessage, "Hint comparison could not be loaded.", error.message);
  }
}

async function loadExperimentD() {
  try {
    const report = await requestJson("/api/experiment-d-judge-package");
    if (!report.available) return;
    renderExperimentD(report);
  } catch (error) {
    setMessage(dHintsMessage, "Experiment D cases could not be loaded.", error.message);
  }
}

accuracyItemSelect.addEventListener("change", renderAccuracyItem);
hintItemSelect.addEventListener("change", () =>
  renderHintPair(hintItems[Number(hintItemSelect.value)], Number(hintItemSelect.value)));
dHintCaseSelect.addEventListener("change", () =>
  renderExperimentDCase(dHintCases[Number(dHintCaseSelect.value)],
    Number(dHintCaseSelect.value)));
runEvaluationButton.addEventListener("click", runEvaluation);
Promise.all([loadAccuracy(), loadHints(), loadExperimentD()]);
