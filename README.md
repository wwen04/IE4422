# Visual evidence-guided science hints

Student research project comparing progressive hints with and without structured visual evidence, using the same frozen vision-language model. No training or fine-tuning.

## Local setup

Use one Python environment for data preparation, model inference and tests. Python 3.12 or newer is required by the dependency pins; Windows Python 3.14.0 is the locally verified environment. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
```

For inference on an NVIDIA GPU, first install the appropriate CUDA-enabled **PyTorch 2.14.1 and torchvision 0.29.1** wheels into this same `.venv`, following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). Use the virtual environment's Python for that command. Then install the shared requirements:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip check
```

`requirements.txt` automatically applies `requirements-lock.txt`; keep both files but do not install them separately. CPU and CUDA PyTorch builds use different wheels, so select the correct build for your machine. CPU wheels support preparation and tests, but the current Qwen engine explicitly requires CUDA for generation. No separate data/Qwen environment is needed. On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python`.

The unified stack retains datasets 5.1.0, Pillow 13.0.0 and huggingface-hub 2.2.0, and uses Transformers 5.21.0. Both datasets and this Transformers release accept Hub versions `>=1.31,<3`. The earlier Transformers 4.57.6 pin required Hub `<1` and was incompatible with the data stack. This was a library-version constraint, not a limitation of the Qwen checkpoint; no dataset-source or preparation-logic change is needed.

The browser UI is the tutor's entry point, locally and through the [Colab notebook and source ZIP](notebooks/README.md). The notebook sets up one project environment that retains Colab's CUDA wheels, prepares the audited dataset, downloads/restores the pinned checkpoint, and launches the same UI. It can save data, checkpoints, and results to Drive. Actual Colab proxy/browser behavior and GPU inference require a real Colab run.

## Prepare ScienceQA

We use only Hugging Face's [`derek-thomas/ScienceQA`](https://huggingface.co/datasets/derek-thomas/ScienceQA). Its supplied train, validation and test splits are preserved. The dataset revision is pinned in `configs/scienceqa.json`, alongside the subject and grade filter. Importing downloads the release once and caches it locally; there is no separate metadata download or manual ZIP setup.

```powershell
.\.venv\Scripts\python prepare_data.py
.\.venv\Scripts\python finalize_data.py
.\.venv\Scripts\python -m unittest discover -s tests -v
```

After cloning the repository, each teammate creates their own environment, runs that install command, then runs `prepare_data.py` and `finalize_data.py` in order. Preparation needs internet access the first time; finalization applies the committed audit decisions to produce the dataset used by the tutor. Later runs reuse the download cache. Changes to the source revision or filters require reviewing audit compatibility before finalization. The last command above runs the optional verification tests.

The first run downloads approximately 626 MB of Parquet files containing text and image bytes. Allow a few GB for the downloaded files, loading cache and exported candidates. No account/token is required for this public dataset. The Hub download cache is stored under `data/cache/huggingface`; the processed Datasets cache is stored under `%LOCALAPPDATA%\IE4422\datasets` on Windows (or `~/.cache/IE4422/datasets` elsewhere) to avoid Windows path-length failures. Set `HF_DATASETS_CACHE` before running to use a different processed-cache location. To reuse the pinned cache without network access:

```powershell
.\.venv\Scripts\python prepare_data.py --offline
```

The cached revision must match the configuration; a different cached release is rejected. A missing cache causes a visible failure.

For filtering/counts and candidate manifests without image decoding or full exports:

```powershell
.\.venv\Scripts\python prepare_data.py --metadata-only --offline
```

Omit `--offline` to allow downloading. **A first metadata-only run can still download the entire release**, because its text and images are stored together. This mode updates candidate manifests and the metadata audit only; it does not refresh existing valid manifests or integrity reports after a filter change. Use a full run before consuming those outputs.

## Automatically evaluate all questions

To automatically run every finalized question across train, validation and test, generating both baseline and project-condition answers and reports, run:

```powershell
.\.venv\Scripts\python evaluate_visual_evidence.py
```

The evaluator fills in model-generated answer choices automatically; no student/demo answer is needed. It compares two answer conditions using the same frozen Qwen model: a direct-answer prompt (default baseline), versus compact, question-focused visual observations followed by an answer prompt that asks the model to verify those observations against the image before solving (project condition). Each question takes three model calls. Results are saved as `outputs/accuracy_comparison.json` (full data), `.csv` (per-question spreadsheet), and `.html` (visual report). It checkpoints each completed question and automatically resumes an interrupted run; use `--no-resume` to discard the saved checkpoint and start over.

The report includes the overall accuracy difference, separate train/validation/test metrics, paired wins/losses, and raw per-question model answers and visual evidence. Open `http://127.0.0.1:8765/compare` for the visual comparison dashboard: it summarizes accuracy, paired outcomes, split results, and per-question answers. `--split validation` evaluates only validation; `--limit 10` evaluates up to the first 10 IDs **per selected split** for a small smoke run. This all-splits run includes test data; treat it as the final evaluation and do not use those test results to tune prompts. A positive difference means higher answer-choice accuracy on these items, not improved hint quality, student learning, or statistical significance. The run requires CUDA and uses answer-key references locally only for scoring; references are never sent to the model.

Alternatively, open `/compare`, choose a validation sample size (1–25), and select **Run comparison**. The dashboard runs the accuracy evaluation and creates paired hint examples in the background, showing progress and errors and refreshing the results when complete. It uses validation data only and needs the configured CUDA model. Each sample size has its own checkpoint so changing the question count does not block a run. Existing hint CSV/key files are reused to protect saved review ratings. Accuracy is scored automatically against the held-out answer references; hint quality remains unrated until a reviewer completes `/review`.

## Review hint quality before using hints with students

Generate a small, answer-key-free validation review sheet with paired direct-visual and evidence-guided hints at three scaffold levels:

```powershell
.\.venv\Scripts\python.exe evaluate_hint_quality.py --limit 10 --seed 4422
```

The script loads only public validation questions and images. It does not send choices or answer-key/reference material to Qwen. It writes `outputs/hint_quality_validation.csv` with blinded A/B conditions and blank reviewer columns for factual accuracy, image grounding, pedagogical usefulness, answer leakage, and unsupported claims. Reviewers should inspect each linked image and question and score every hint level. Use the same 1–4 scale for the three rating columns: **1** = incorrect/unsafe or not useful; **2** = major issue; **3** = mostly sound/useful with a minor issue; **4** = accurate, well-grounded, and useful. Mark answer leakage `yes` if the hint gives away the answer, option, or final result; mark unsupported claim `yes` if it asserts a detail not supported by the question/image. Keep the matching `_key.json` private until reviews are complete; it maps A/B to generation method and stores model-extracted evidence. It contains no answer key. A repeatable sample can be generated with the same seed; use a different seed for a separate sample.

After completing the rating columns, summarize the sheet without loading the model:

```powershell
.\.venv\Scripts\python.exe evaluate_hint_quality.py --summarize outputs/hint_quality_validation.csv outputs/hint_quality_validation_key.json
```

For a visual reviewer workflow integrated with the local Fieldnotes interface, start `ui_server.py` and open `http://127.0.0.1:8765/review` (or choose **Hint review studio** from the tutor page). The reviewer view presents the question, context, choices, image, and anonymized A/B hint sets side by side. Ratings and notes can be saved back into the existing CSV as you move between questions. The page/API do not expose the A/B mapping key. Keep the `_key.json` file private until all reviewers have submitted their ratings; summarize the saved CSV afterward with the command above.

After generating the hint set, use `http://127.0.0.1:8765/compare` to inspect the paired no-evidence/evidence-guided hints, the model-generated visual evidence, and any saved human ratings. This research page intentionally reveals which condition is which; use `/review` for blinded scoring and do not share `/compare` or its API with reviewers before ratings are complete. Accuracy results and hint ratings are separate outcomes and are shown in separate sections. Missing reports are identified in the interface with the command needed to create them.

This reports human hint-quality ratings only. It does not establish that students answer more accurately or learn more. Any student study needs appropriate instructor/ethics approval, informed consent, randomized comparable conditions, and an unassisted follow-up if measuring learning. Do not treat unreviewed Qwen hints as verified or safe for students.

## Flow and identifiers

`prepare_data.py` loads the pinned release once, audits the filter on its rows, validates candidates, then exports the usable questions and their images. Original row positions are retained before filtering or validation. IDs such as `train_000042` mean zero-based row 42 of the pinned training split; they are local project IDs, not the author's question IDs. Adding or removing other questions through filtering does not renumber retained rows. Changing the dataset revision requires regenerating and reviewing selections.

The candidate filter is **image present + subject natural science + Grade 7 or 8**. The current pinned release reproduces **1,272 train / 421 validation / 382 test candidates**, all passing validation. These counts are a regression baseline rather than a runtime restriction on intentional filter changes.

Validation checks questions, ordered choices, answer indices, required field types and full image readability. Invalid candidates are recorded with reasons. Source/schema/revision errors stop processing. Exported images retain the original bytes without resizing.

ScienceQA's `hint` field is supplied problem context and becomes `context` in our public records. Answers, lectures and solutions are written to separate reference files. Future generation code must load only public questions and build prompts from allowed fields; evaluation can join references by `item_id`.

Image presence alone does not establish that a question needs visual reasoning. The separate [assisted visual review](reviews/scienceqa/README.md) now records this judgement for each prepared candidate. Its derived selection is separate from preparation and is not independently adjudicated human ground truth. No tutor outputs or reference answers were used for this selection.

## Prepare the AI2D evaluation sample

AI2D is a separate, frozen **100-question test sample for evaluation without student attempts**. It is not added to the browser tutor or combined with ScienceQA scores. The source is pinned in `configs/ai2d.json`; source downloads use the existing `data/cache/huggingface` cache.

```powershell
.\.venv\Scripts\python.exe prepare_ai2d.py
# Rebuild from the same cached source without network access:
.\.venv\Scripts\python.exe prepare_ai2d.py --offline
```

Preparation validates all source questions, nonempty choices, answer positions and readable images. It preserves question text, choice text/order/case and original image bytes. It assigns A-D option IDs to this source's four choices and maps the source answer index to the option ID; it does not change labels inside diagrams. The common format accepts 2-26 choices. Missing lecture and solution fields are exported as empty strings with an explicit unavailable flag in the separate references.

Questions sharing a source image filename or identical decoded pixels form one image group. Using seed 42, the script randomly selects 100 groups and one question per group. It exports all usable rows under `data/processed/ai2d/` and the frozen sample under `data/final/ai2d/`, each with separate `questions/test.jsonl`, `references/test.jsonl` and original image files. It preserves the source test split and assigns IDs such as `ai2d_test_000042` from original row positions.

The committed selection lives in `manifests/ai2d/test_final_ids.json` and `test_selection.json`, alongside the all-valid ID list. `reports/ai2d/` records counts, exclusions, checksums, sampling and limitations. Rebuilding must reproduce the frozen configuration, source fingerprints and selection; incompatible inputs fail before replacing final exports. This selection has **no LLM visual-necessity review or manual label-matching pass**. See [the selection policy](reviews/ai2d/README.md) and [preparation report](reports/ai2d/preparation.md). Experiments and judge packages are separate later steps.

## Inspect questions and apply the visual-need audit

The separate [visual-need audit](reviews/scienceqa/README.md) records assisted decisions for all 2,075 prepared Grade 7-8 candidates. It leaves preparation outputs intact and exports only image-required rows under `data/final/scienceqa/`, with additional `<split>_final_ids.json` manifests.

```powershell
.\.venv\Scripts\python.exe inspect_question.py validation_000008 --open
.\.venv\Scripts\python.exe finalize_data.py --check-only
.\.venv\Scripts\python.exe finalize_data.py
```

The inspector defaults to audited final questions and images. It generates a replaceable Markdown page and matching browser page containing the question, context, choices, image, metadata, audit classification and separate reference material. Omit the ID to be prompted, use `--list validation` to list final IDs, or set `--output outputs/inspection/example.md` to keep a separate page. To inspect an excluded candidate, explicitly use `--source processed`, for example `inspect_question.py validation_000003 --source processed --open`. This inspection mode does not change tutor inputs. It needs no model or GPU. Completed audit decisions are stored outside `data/` and survive regeneration of the page or prepared data. See [the audit report](reports/scienceqa/visual_audit.md) for counts and boundaries. The browser tutor, inspector and Colab UI use the audited final set; missing final data causes an error instead of a fallback to candidates.

## Browser interface

After installing the requirements and preparing ScienceQA, start the local browser UI from the repository root:

```powershell
.\.venv\Scripts\python ui_server.py
```

The app opens at `http://127.0.0.1:8765`. Before starting the server, startup checks that prepared question/reference exports, valid-ID manifests and the integrity report exist; if any are missing, it runs `prepare_data.py`. It then runs `finalize_data.py` to validate the saved visual-need audit and rebuild the final selection. The UI displays only finalized questions marked as requiring visual evidence. When a question is selected, the UI runs `inspect_question.py` for that item. **Show inspection report** optionally displays the generated report inside the UI; it includes the audit rationale and answer/reference material, so it stays hidden unless explicitly opened. The first preparation needs internet access to download ScienceQA; later runs use the local caches. If preparation or finalization fails, startup stops and reports the error instead of opening an incomplete UI. Choose **I'm stuck — give me a hint** without entering an answer, or submit a choice and explanation with **Check my answer** to get feedback or a follow-up hint. **Load demo response** fills clearly marked example inputs. Model outputs are experimental and may be incorrect; they are not a formal assessment. Each completed run is saved under `outputs/`. Stop the local server with Ctrl+C.

The server binds to localhost only and uses Python's standard library; it adds no web framework dependency. Tutor generation still requires the NVIDIA CUDA setup described above.

## Tutor workflow

`ui_server.py` is the only interactive tutor entry point. It imports `scienceqa_tutor.py` for dataset loading and prompt formatting, `tutor_pipeline.py` for the two tutor branches, and `qwen_engine.py` for the shared model. `hint_prompts.py` and `prompts/initial_hint.txt` provide the no-attempt hint wording.

Every model call receives the image plus the question, supplied context when present, and ordered choices. Requesting a hint ignores any attempt currently entered in the form and makes two model calls: evidence extraction, then one hint at the selected tier.

Checking an answer extracts evidence, compares the selected choice against the gold option ID in Python, and asks the model to assess the student's reasoning using the image, public problem inputs, selected choice, reasoning, and reference lecture/solution. The assessment does not receive the extracted evidence. A correct choice with supported reasoning returns fixed positive feedback after two model calls. Otherwise, diagnosis receives the public inputs, attempt, evidence, and misconception categories; hint generation receives the public inputs, evidence, diagnosis, and tier. That branch takes four model calls. Reference material is supplied only to the reasoning assessment; feedback messages are fixed Python strings.

The server loads one model lazily and reuses it across requests. Each generation call is a fresh image-and-text message; previous hints and attempts are not added as conversation history. Evidence is requested as at most 10 concise numbered observations, each with a type and location, under the existing 320-token limit. The shared evidence prompt uses the public question, context and options only to identify relevant visual details; options are not evidence. It asks for the minimum sufficient observations and explicitly excludes inference, calculations, answering and hint generation. Evidence is stored as raw text without format or correctness validation, repair or caching. Diagnosis is still requested as JSON without schema validation or caching. Each completed request is saved as `outputs/ui_<job-id>.json`.

Separate research scripts and protocols remain under `experiments/`. They are outside the UI workflow and are excluded from the Colab UI notebook and source ZIP. The separate [batch-experiment guide](experiments/README.md) covers AI2D/ScienceQA A, B and D, the frozen 30-question AI2D judging sample and per-dataset summaries. Use [IE4422_Experiments.ipynb](notebooks/IE4422_Experiments.ipynb) with `IE4422_Experiments_Source.zip` for Colab batch runs; the browser-tutor notebook remains separate.

## Folders and GitHub teamwork

```text
prepare_data.py                         pinned dataset preparation
finalize_data.py                        apply saved audit decisions
inspect_question.py                     inspect final questions; explicit candidate inspection option
qwen_engine.py                          shared image/text model adapter
tutor_pipeline.py                       prompt stages using a supplied tutor
run_scienceqa_tutor.py                   prepared-question CLI and Python helper
evaluate_visual_evidence.py              resumable all-split accuracy evaluation and reports
run_tutor.py / inspect_diagram.py        optional custom-diagram entry points
scienceqa_tutor.py                       shared audited-question loaders and UI helpers
ui_server.py                            browser tutor entry point
hint_prompts.py / prompts/              shared hint wording
experiments/                            separate research scripts and protocols; excluded from UI bundle
requirements.txt / requirements-lock.txt shared environment and constraints
notebooks/                              UI-only Colab notebook, source ZIP and helpers
configs/ / tests/ / docs/               configuration, checks and review guidance
data/                                    ignored by Git
  cache/huggingface/                      Hugging Face Hub download cache
  processed/scienceqa/
    images/train_000042.png               local-ID filenames, original image bytes
    questions/<split>.jsonl               public inputs, one JSON object per line
    references/<split>.jsonl              gold option, lecture and solution
    visual_review_template.csv            blank worksheet for human review
    environment.json                     Python and main package versions
  final/scienceqa/
    images/<item_id>.png                  audited image copies used by the tutor
    questions/<split>.jsonl               retained public questions
    references/<split>.jsonl              matching separate reference answers
    selection.json                       audit provenance and final counts
reviews/scienceqa/                        committed audit decisions and provenance
manifests/scienceqa/                      small ID lists to commit
  <split>_candidate_ids.json              passed the subject/grade/image-presence filter
  <split>_valid_ids.json                  also passed question and image validation
  <split>_final_ids.json                  retained by the visual-need audit
reports/scienceqa/                        generated audit results to commit
  metadata_audit.json                     source, ID convention and filter counts
  integrity_audit.json                    validation counts, exclusions and overlap groups
```

Commit scripts, tests, configuration, dependency pins, ID manifests and audit reports. Each teammate runs the same command locally. `.gitignore` excludes all `data/`, virtual environments, model weights, downloaded Qwen examples, generated outputs, temporary files and local secrets. The implementation plan, agent instructions, proposal and project diagram are local-only and excluded as well.

Candidate/valid manifests document preparation; `prepare_data.py` selects using source-row indices. `finalize_data.py` applies the saved review decisions. Runtime question loaders require final exports to match the final-ID manifests and require images under `data/final/scienceqa/images/`. Reports describe preparation and curation, not hint-quality experiment results.

A full preparation run overwrites preparation manifests, reports, question/reference exports and the blank review template. Saved decisions live separately under `reviews/`. Run finalization afterwards before using the tutor. Old image files do not define membership: final manifests and final question exports define the runtime selection.

Exact cross-split overlaps are checked by directly comparing question/context/ordered-choice values and decoded RGB pixels with matching dimensions. Pixel comparisons process one size group at a time. Reports contain matching IDs, with no automatic removal or split reassignment. These checks do not cover paraphrases, near-duplicate images or pretrained-model contamination. Shared images can support different questions; agree on an overlap policy before final sampling.

## Dataset attribution

Lu et al. (2022), *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*, NeurIPS. [Paper](https://arxiv.org/abs/2209.09513), [project](https://scienceqa.github.io/).

The upstream dataset documentation and the loader included with the Hugging Face release specify **CC BY-NC-SA 4.0**. The Hugging Face card's top-level tag instead says CC BY-SA 4.0; retain the upstream attribution and noncommercial/share-alike terms when sharing dataset-derived content. The dataset is downloaded locally rather than bundled with the code repository.
