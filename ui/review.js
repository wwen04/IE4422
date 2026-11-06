const progressLabel = document.querySelector("#review-progress-label");
const progressBar = document.querySelector("#review-progress-bar");
const reviewerName = document.querySelector("#reviewer-name");
const previousButton = document.querySelector("#previous-button");
const saveButton = document.querySelector("#save-button");
const nextButton = document.querySelector("#next-button");
const workspace = document.querySelector("#review-workspace");
const reviewMessage = document.querySelector("#review-message");
const reviewError = document.querySelector("#review-error");
const conditionCards = document.querySelector("#condition-cards");
const commentsInput = document.querySelector("#review-comments");

const ratingDimensions = [
  ["factual_accuracy_1_to_4", "Factual accuracy", ["1", "2", "3", "4"]],
  ["image_groundedness_1_to_4", "Grounded in image", ["1", "2", "3", "4"]],
  ["pedagogical_usefulness_1_to_4", "Useful as a hint", ["1", "2", "3", "4"]],
  ["answer_leakage_yes_no_unclear", "Gives away answer?", ["no", "yes", "unclear"]],
  ["unsupported_claim_yes_no", "Unsupported claim?", ["no", "yes"]],
];

let items = [];
let currentIndex = 0;
let dirty = false;
let busy = false;

function show(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}

function currentItem() {
  return items[currentIndex];
}

function isConditionComplete(condition) {
  return [1, 2, 3].every((level) =>
    ratingDimensions.every(([dimension]) =>
      Boolean(condition.ratings[`level_${level}_${dimension}`])
    )
  );
}

function updateProgress() {
  const completed = items.filter((item) =>
    item.conditions.length === 2 && item.conditions.every(isConditionComplete)
  ).length;
  const total = items.length;
  progressLabel.textContent = `${completed} of ${total} questions fully rated`;
  progressBar.style.width = total ? `${completed / total * 100}%` : "0%";
  progressBar.parentElement.setAttribute("aria-valuenow", String(completed));
  progressBar.parentElement.setAttribute("aria-valuemax", String(total));
}

function setDirty(value = true) {
  dirty = value;
  saveButton.textContent = dirty ? "Save ratings •" : "Ratings saved";
}

function makeRatingField(item, condition, level, dimension, title, values) {
  const fieldset = document.createElement("fieldset");
  fieldset.className = "rating-field";
  const legend = document.createElement("legend");
  legend.textContent = title;
  fieldset.append(legend);

  const options = document.createElement("div");
  options.className = "rating-options";
  const name = `${item.item_id}-${condition.condition_code}-level-${level}-${dimension}`;
  for (const value of values) {
    const label = document.createElement("label");
    label.className = "rating-option";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = name;
    input.value = value;
    input.checked = condition.ratings[`level_${level}_${dimension}`] === value;
    input.addEventListener("change", () => {
      condition.ratings[`level_${level}_${dimension}`] = value;
      setDirty();
      updateProgress();
      show(reviewMessage, "");
    });
    const text = document.createElement("span");
    text.className = values.length > 2 ? "" : "rating-label";
    text.textContent = value === "unclear" ? "?" : value;
    label.append(input, text);
    options.append(label);
  }
  fieldset.append(options);
  return fieldset;
}

function renderCondition(item, condition) {
  const card = document.createElement("article");
  card.className = "condition-card";
  const heading = document.createElement("header");
  heading.className = "condition-heading";
  const title = document.createElement("h3");
  title.textContent = "Anonymized hint set";
  const code = document.createElement("span");
  code.className = "condition-code";
  code.textContent = condition.condition_code;
  code.setAttribute("aria-label", `Condition ${condition.condition_code}`);
  heading.append(title, code);
  card.append(heading);

  for (let level = 1; level <= 3; level += 1) {
    const section = document.createElement("section");
    section.className = "hint-level-review";
    const titleRow = document.createElement("div");
    titleRow.className = "hint-level-title";
    const levelTitle = document.createElement("h4");
    levelTitle.textContent = `Hint level ${level}`;
    const levelNote = document.createElement("span");
    levelNote.textContent = ["NOTICE", "CONNECT", "REASON"][level - 1];
    titleRow.append(levelTitle, levelNote);

    const hint = document.createElement("p");
    hint.className = "hint-copy";
    hint.textContent = condition.hints[level - 1] || "(No hint was generated.)";
    const grid = document.createElement("div");
    grid.className = "rating-grid";
    for (const [dimension, label, values] of ratingDimensions) {
      grid.append(makeRatingField(item, condition, level, dimension, label, values));
    }
    section.append(titleRow, hint, grid);
    card.append(section);
  }
  return card;
}

function renderItem() {
  const item = currentItem();
  if (!item) return;
  document.querySelector("#question-position").textContent =
    `${String(currentIndex + 1).padStart(2, "0")} / ${items.length}`;
  document.querySelector("#review-question").textContent = item.question;
  const context = document.querySelector("#review-context");
  context.textContent = item.context;
  context.hidden = !item.context;

  const choices = document.querySelector("#review-choices");
  choices.replaceChildren();
  for (const choice of item.choices.split("\n").filter(Boolean)) {
    const row = document.createElement("div");
    row.className = "choice-item";
    const marker = document.createElement("span");
    marker.className = "choice-letter";
    marker.textContent = choice.split(".", 1)[0];
    const text = document.createElement("span");
    text.textContent = choice.slice(choice.indexOf(".") + 1).trim();
    row.append(marker, text);
    choices.append(row);
  }

  const image = document.querySelector("#review-image");
  image.src = item.image_url;
  image.alt = `Diagram for ${item.item_id}`;
  document.querySelector("#image-link").href = item.image_url;
  commentsInput.value = item.conditions[0]?.comments || "";
  conditionCards.replaceChildren(
    ...item.conditions.map((condition) => renderCondition(item, condition))
  );
  previousButton.disabled = busy || currentIndex === 0;
  nextButton.disabled = busy || currentIndex === items.length - 1;
  saveButton.disabled = busy;
  setDirty(false);
  updateProgress();
}

async function saveCurrent() {
  if (!dirty) {
    show(reviewMessage, "All changes on this question are saved.");
    return true;
  }
  busy = true;
  previousButton.disabled = true;
  nextButton.disabled = true;
  saveButton.disabled = true;
  show(reviewMessage, "Saving your ratings…");
  show(reviewError, "");
  const item = currentItem();
  try {
    for (const condition of item.conditions) {
      const payload = {
        item_id: item.item_id,
        condition_code: condition.condition_code,
        ratings: condition.ratings,
        reviewer: reviewerName.value.trim(),
        comments: commentsInput.value,
      };
      await requestJson("/api/hint-reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      condition.reviewer = payload.reviewer;
      condition.comments = payload.comments;
    }
    setDirty(false);
    show(reviewMessage, "Ratings saved to the review CSV on this computer.");
    updateProgress();
    return true;
  } catch (error) {
    show(reviewError, `Could not save ratings: ${error.message}`);
    return false;
  } finally {
    busy = false;
    previousButton.disabled = currentIndex === 0;
    nextButton.disabled = currentIndex === items.length - 1;
    saveButton.disabled = false;
  }
}

async function move(offset) {
  if (dirty && !(await saveCurrent())) return;
  currentIndex = Math.max(0, Math.min(items.length - 1, currentIndex + offset));
  show(reviewMessage, "");
  show(reviewError, "");
  renderItem();
}

async function start() {
  try {
    const data = await requestJson("/api/hint-reviews");
    items = data.items;
    if (!items.length) {
      show(reviewError, "The review sheet contains no questions.");
      progressLabel.textContent = "No review items available";
      return;
    }
    workspace.hidden = false;
    renderItem();
  } catch (error) {
    progressLabel.textContent = "Review set unavailable";
    show(reviewError, error.message);
  }
}

reviewerName.addEventListener("input", () => setDirty());
commentsInput.addEventListener("input", () => setDirty());
saveButton.addEventListener("click", saveCurrent);
previousButton.addEventListener("click", () => move(-1));
nextButton.addEventListener("click", () => move(1));
window.addEventListener("beforeunload", (event) => {
  if (!dirty) return;
  event.preventDefault();
  event.returnValue = "";
});

start();
