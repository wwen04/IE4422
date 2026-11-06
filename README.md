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

Dependency consistency, imports and preparation/tests are checked locally. GPU inference and the research quality of outputs still need a feasibility trial. Teammates without a local GPU can use the [Colab notebook and source ZIP](notebooks/README.md). They run the same model/prompt functions in a single Colab project environment that retains Colab's CUDA wheels; actual assigned-hardware feasibility still needs a real Colab run.

## Prepare ScienceQA

We use only Hugging Face's [`derek-thomas/ScienceQA`](https://huggingface.co/datasets/derek-thomas/ScienceQA). Its supplied train, validation and test splits are preserved. The dataset revision is pinned in `configs/scienceqa.json`, alongside the subject and grade filter. Importing downloads the release once and caches it locally; there is no separate metadata download or manual ZIP setup.

```powershell
.\.venv\Scripts\python prepare_data.py
.\.venv\Scripts\python -m unittest discover -s tests -v
```

After cloning the repository, each teammate creates their own environment, runs that install command, then runs `prepare_data.py` once with internet access. This recreates their ignored local dataset and the shared manifests/reports. Later runs reuse the download cache; run preparation again when the source revision or filters change. The final command above runs the optional verification tests.

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

The report includes the overall accuracy difference, separate train/validation/test metrics, paired wins/losses, and raw per-question model answers and visual evidence. The UI shows the comparison and provides links to the visual report and JSON. `--split validation` evaluates only validation; `--limit 10` evaluates up to the first 10 IDs **per selected split** for a small smoke run. This all-splits run includes test data; treat it as the final evaluation and do not use those test results to tune prompts. A positive difference means higher answer-choice accuracy on these items, not improved hint quality, student learning, or statistical significance. The run requires CUDA and uses answer-key references locally only for scoring; references are never sent to the model.

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

This reports human hint-quality ratings only. It does not establish that students answer more accurately or learn more. Any student study needs appropriate instructor/ethics approval, informed consent, randomized comparable conditions, and an unassisted follow-up if measuring learning. Do not treat unreviewed Qwen hints as verified or safe for students.

## Flow and identifiers

`prepare_data.py` loads the pinned release once, audits the filter on its rows, validates candidates, then exports the usable questions and their images. Original row positions are retained before filtering or validation. IDs such as `train_000042` mean zero-based row 42 of the pinned training split; they are local project IDs, not the author's question IDs. Adding or removing other questions through filtering does not renumber retained rows. Changing the dataset revision requires regenerating and reviewing selections.

The candidate filter is **image present + subject natural science + Grade 7 or 8**. The current pinned release reproduces **1,272 train / 421 validation / 382 test candidates**, all passing validation. These counts are a regression baseline rather than a runtime restriction on intentional filter changes.

Validation checks questions, ordered choices, answer indices, required field types and full image readability. Invalid candidates are recorded with reasons. Source/schema/revision errors stop processing. Exported images retain the original bytes without resizing.

ScienceQA's `hint` field is supplied problem context and becomes `context` in our public records. Answers, lectures and solutions are written to separate reference files. Future generation code must load only public questions and build prompts from allowed fields; evaluation can join references by `item_id`.

Image presence alone does not establish that a question needs visual reasoning. The separate [assisted visual review](reviews/scienceqa/README.md) now records this judgement for each prepared candidate. Its derived selection is separate from preparation and is not independently adjudicated human ground truth. No tutor outputs or reference answers were used for this selection.

## Inspect questions and apply the visual-need audit

The separate [visual-need audit](reviews/scienceqa/README.md) records assisted decisions for all 2,075 prepared Grade 7-8 candidates. It leaves preparation outputs intact and exports only image-required rows under `data/final/scienceqa/`, with additional `<split>_final_ids.json` manifests.

```powershell
.\.venv\Scripts\python.exe inspect_question.py validation_000003 --open
.\.venv\Scripts\python.exe finalize_data.py --check-only
.\.venv\Scripts\python.exe finalize_data.py
```

The inspector generates a replaceable Markdown page and matching browser page containing the question, context, choices, image, metadata, audit classification and separate reference material. Omit the ID to be prompted, use `--list validation` to list IDs, or set `--output outputs/inspection/example.md` to keep a separate page. It needs no model or GPU. Completed audit decisions are stored outside `data/` and survive regeneration of the page or prepared data. See [the audit report](reports/scienceqa/visual_audit.md) for counts and boundaries. The tutor commands below still use the original prepared data.

## Run the tutor locally

Check that the selected Python can use CUDA:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"
```

Choose a prepared validation ID and provide the student's actual response and reasoning. Replace the example explanation before running:

```powershell
.\.venv\Scripts\python run_scienceqa_tutor.py --split validation --item-id validation_000003 --answer "A" --explanation "Replace with the student's own reasoning" --hint-level 1
```

The first run downloads `Qwen/Qwen2.5-VL-3B-Instruct`; use `--model-path` to select a complete existing checkpoint folder. The engine loads it in 4-bit and uses deterministic generation. The default result is `outputs/scienceqa_tutor.json`; choose a distinct `--output` path to retain multiple runs. The runner reads public questions only and preserves IDs, supplied context, option order and image files. It does not load reference exports.

This remains an exploratory prototype. The terminal entry point generates a response-informed hint. The browser UI additionally supports a pre-answer, answer-independent hint. Once an answer is submitted, the UI compares its choice ID with the separate reference export; if the choice matches, the model checks whether the explanation is supported by the reference lecture/solution. A correct answer with supported reasoning gets positive feedback; an incorrect answer or unclear/weak explanation gets another hint. The reference answer is loaded only for a submitted check and is not included in the question API or displayed as the answer key. Model assessments can still be wrong; they are not verified annotations or a formal grade. This does not implement the full research methodology: evidence validation/caching, three-hint history, disclosure checking, A–E comparisons and human evaluation remain pending. Development uses train/validation; preserve test items for the agreed final evaluation.

## Browser interface

After installing the requirements and preparing ScienceQA, start the local browser UI from the repository root:

```powershell
.\.venv\Scripts\python ui_server.py
```

The app opens at `http://127.0.0.1:8765`. Before starting the server, startup checks that prepared question/reference exports, valid-ID manifests and the integrity report exist; if any are missing, it runs `prepare_data.py`. It then runs `finalize_data.py` to validate the saved visual-need audit and rebuild the final selection. The UI displays only finalized questions marked as requiring visual evidence. When a question is selected, the UI runs `inspect_question.py` for that item. **Show inspection report** optionally displays the generated report inside the UI; it includes the audit rationale and answer/reference material, so it stays hidden unless explicitly opened. The first preparation needs internet access to download ScienceQA; later runs use the local caches. If preparation or finalization fails, startup stops and reports the error instead of opening an incomplete UI. Choose **I'm stuck — give me a hint** without entering an answer, or submit a choice and explanation with **Check my answer** to get feedback or a follow-up hint. **Load demo response** fills clearly marked example inputs. Model outputs are experimental and may be incorrect; they are not a formal assessment. Each completed run is saved under `outputs/`. Stop the local server with Ctrl+C.

The server binds to localhost only and uses Python's standard library; it adds no web framework dependency. Tutor generation still requires the NVIDIA CUDA setup described above.

For custom diagrams, the optional root entry points are:

```powershell
.\.venv\Scripts\python inspect_diagram.py --image "C:\path\to\diagram.png"
.\.venv\Scripts\python run_tutor.py --image "C:\path\to\diagram.png" --question "Your matching question" --answer "Student response" --explanation "Student reasoning" --hint-level 1
```

These paths must refer to a real image. `inspect_diagram.py` reports visible features; `run_tutor.py` saves a custom-question result under `outputs/`. All three entry points support `--help` without loading a model.

### Reuse one loaded model in Python

```python
from qwen_engine import QwenTutor
from run_scienceqa_tutor import ROOT, load_question, run_question

tutor = QwenTutor()  # Load once for this Python session.
item = load_question(ROOT, "validation", "validation_000003")
result = run_question(tutor, item, "A", "The student's actual reasoning", hint_level=1)
# Call run_question(tutor, ...) again for other items or hint levels.
```

The lower-level `tutor_pipeline.run_pipeline(tutor, image, question, student_answer, explanation, hint_level=1)` also accepts the existing model. Neither function constructs a model. Separate terminal invocations are separate processes and load their own model. This change reuses model weights in memory; evidence and diagnosis are still regenerated on each pipeline call.

## Folders and GitHub teamwork

```text
prepare_data.py                         pinned dataset preparation
qwen_engine.py                          shared image/text model adapter
tutor_pipeline.py                       prompt stages using a supplied tutor
run_scienceqa_tutor.py                   prepared-question CLI and Python helper
evaluate_visual_evidence.py              resumable all-split accuracy evaluation and reports
run_tutor.py / inspect_diagram.py        optional custom-diagram entry points
requirements.txt / requirements-lock.txt shared environment and constraints
notebooks/                              Colab notebook, source ZIP and helpers
configs/ / tests/ / docs/               configuration, checks and review guidance
data/                                    ignored by Git
  cache/huggingface/                      Hugging Face Hub download cache
  processed/scienceqa/
    images/train_000042.png               local-ID filenames, original image bytes
    questions/<split>.jsonl               public inputs, one JSON object per line
    references/<split>.jsonl              gold option, lecture and solution
    visual_review_template.csv            blank worksheet for human review
    environment.json                     Python and main package versions
manifests/scienceqa/                      small ID lists to commit
  <split>_candidate_ids.json              passed the subject/grade/image-presence filter
  <split>_valid_ids.json                  also passed question and image validation
reports/scienceqa/                        generated audit results to commit
  metadata_audit.json                     source, ID convention and filter counts
  integrity_audit.json                    validation counts, exclusions and overlap groups
```

Commit scripts, tests, configuration, dependency pins, ID manifests and audit reports. Each teammate runs the same command locally. `.gitignore` excludes all `data/`, virtual environments, model weights, downloaded Qwen examples, generated outputs, temporary files and local secrets. The implementation plan, agent instructions, proposal and project diagram are local-only and excluded as well.

Manifests document the shared selection; the current script selects using source-row indices held in memory rather than reading the manifests back. Reports describe dataset preparation, not hint-quality experiment results.

A full preparation run overwrites generated manifests, reports, question/reference exports and the blank review template. Save completed reviews separately before rerunning. If changing filters, do not treat old files left in `images/` as the selection: current valid manifests and question exports define the usable set.

Exact cross-split overlaps are checked by directly comparing question/context/ordered-choice values and decoded RGB pixels with matching dimensions. Pixel comparisons process one size group at a time. Reports contain matching IDs, with no automatic removal or split reassignment. These checks do not cover paraphrases, near-duplicate images or pretrained-model contamination. Shared images can support different questions; agree on an overlap policy before final sampling.

## Dataset attribution

Lu et al. (2022), *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*, NeurIPS. [Paper](https://arxiv.org/abs/2209.09513), [project](https://scienceqa.github.io/).

The upstream dataset documentation and the loader included with the Hugging Face release specify **CC BY-NC-SA 4.0**. The Hugging Face card's top-level tag instead says CC BY-SA 4.0; retain the upstream attribution and noncommercial/share-alike terms when sharing dataset-derived content. The dataset is downloaded locally rather than bundled with the code repository.
