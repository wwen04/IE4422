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

Dependency consistency, imports and preparation/tests are checked locally. GPU inference and the research quality of outputs still need a feasibility trial. Teammates without a local GPU can use the [Colab notebook and source ZIP](notebooks/README.md). The notebook defaults to the answer-only training checker and can save results/checkpoints to Drive. The browser UI and original three-hint demo remain optional. It uses a single project environment that retains Colab's CUDA wheels; see the notebook guide for Google's free-runtime web-UI restrictions. Actual proxy/browser behavior and GPU feasibility still need a real Colab run.

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

## Flow and identifiers

`prepare_data.py` loads the pinned release once, audits the filter on its rows, validates candidates, then exports the usable questions and their images. Original row positions are retained before filtering or validation. IDs such as `train_000042` mean zero-based row 42 of the pinned training split; they are local project IDs, not the author's question IDs. Adding or removing other questions through filtering does not renumber retained rows. Changing the dataset revision requires regenerating and reviewing selections.

The candidate filter is **image present + subject natural science + Grade 7 or 8**. The current pinned release reproduces **1,272 train / 421 validation / 382 test candidates**, all passing validation. These counts are a regression baseline rather than a runtime restriction on intentional filter changes.

Validation checks questions, ordered choices, answer indices, required field types and full image readability. Invalid candidates are recorded with reasons. Source/schema/revision errors stop processing. Exported images retain the original bytes without resizing.

ScienceQA's `hint` field is supplied problem context and becomes `context` in our public records. Answers, lectures and solutions are written to separate reference files. Future generation code must load only public questions and build prompts from allowed fields; evaluation can join references by `item_id`.

Image presence alone does not establish that a question needs visual reasoning. The separate [assisted visual review](reviews/scienceqa/README.md) now records this judgement for each prepared candidate. Its derived selection is separate from preparation and is not independently adjudicated human ground truth. No tutor outputs or reference answers were used for this selection.

## Inspect questions and apply the visual-need audit

The separate [visual-need audit](reviews/scienceqa/README.md) records assisted decisions for all 2,075 prepared Grade 7-8 candidates. It leaves preparation outputs intact and exports only image-required rows under `data/final/scienceqa/`, with additional `<split>_final_ids.json` manifests.

```powershell
.\.venv\Scripts\python.exe inspect_question.py validation_000008 --open
.\.venv\Scripts\python.exe finalize_data.py --check-only
.\.venv\Scripts\python.exe finalize_data.py
```

The inspector defaults to audited final questions and images. It generates a replaceable Markdown page and matching browser page containing the question, context, choices, image, metadata, audit classification and separate reference material. Omit the ID to be prompted, use `--list validation` to list final IDs, or set `--output outputs/inspection/example.md` to keep a separate page. To inspect an excluded candidate, explicitly use `--source processed`, for example `inspect_question.py validation_000003 --source processed --open`. This inspection mode does not change tutor inputs. It needs no model or GPU. Completed audit decisions are stored outside `data/` and survive regeneration of the page or prepared data. See [the audit report](reports/scienceqa/visual_audit.md) for counts and boundaries. The CLI, Python helper, browser tutor and Colab demo use the audited final set; missing final data causes an error instead of a fallback to candidates.

## Answer-only training check

`check_scienceqa_answers.py` is a standalone command for the answer-only development check. It uses only the current audited final **training** selection (natural science, Grades 7–8, image required by the saved audit). It does not train or update the model. Validation/test questions cannot be selected. Dataset preparation still exports the official splits for the rest of the project.

After preparing/finalizing data and setting up a CUDA GPU, check the first ten retained training questions:

```powershell
.\.venv\Scripts\python check_scienceqa_answers.py --limit 10
```

Omit `--limit` to check the whole audited training set. To select particular retained training questions, use `--item-id train_000006` and repeat `--item-id` for more IDs. `--model-path` accepts an existing checkpoint folder; the default is the same `Qwen/Qwen2.5-VL-3B-Instruct` model used by the tutor. `--help` needs no GPU or model download.

One loaded model serves all selected questions, with one deterministic call per question. Its inputs are the image, question, supplied context and ordered options. The prompt requests exactly one option letter without a justification. There is no evidence extraction, hint generation or learner-attempt assessment. The checker loads only training reference letters after all generation calls, then compares option IDs directly. Reference lectures/solutions never enter prompts.

Each run creates a new folder under `outputs/answer_checks/`; `--output-dir` can instead name a new folder. `predictions.jsonl` is flushed after each question and retains raw responses, predictions, timing and errors. `run.json` records selected IDs, model/generation settings and run status. `result.json` adds reference letters, correctness and summary counts. Whitespace and letter case are normalized; extra text, empty responses and letters outside the supplied options are invalid. Invalid outputs and generation failures count as unsuccessful in overall accuracy. A random-guess baseline accounts for varying option counts. Missing or invalid references stop scoring rather than shrink the denominator.

These training results are development feedback, not held-out accuracy or evidence of hint quality. This answer-only check is narrower than the proposal's answer-and-explanation condition A. The implementation has been checked with fake model responses; real model results have not been run.

## Run the tutor locally

Check that the selected Python can use CUDA:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"
```

Choose an audited final validation ID and provide the student's actual response and reasoning. Replace the example explanation before running:

```powershell
.\.venv\Scripts\python run_scienceqa_tutor.py --split validation --item-id validation_000008 --answer "A" --explanation "Replace with the student's own reasoning" --hint-level 1
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
item = load_question(ROOT, "validation", "validation_000008")
result = run_question(tutor, item, "A", "The student's actual reasoning", hint_level=1)
# Call run_question(tutor, ...) again for other items or hint levels.
```

The lower-level `tutor_pipeline.run_pipeline(tutor, image, question, student_answer, explanation, hint_level=1)` also accepts the existing model. Neither function constructs a model. Separate terminal invocations are separate processes and load their own model. This change reuses model weights in memory; evidence and diagnosis are still regenerated on each pipeline call.

## Folders and GitHub teamwork

```text
prepare_data.py                         pinned dataset preparation
finalize_data.py                        apply saved audit decisions
inspect_question.py                     inspect final questions; explicit candidate inspection option
qwen_engine.py                          shared image/text model adapter
tutor_pipeline.py                       prompt stages using a supplied tutor
run_scienceqa_tutor.py                   audited-question CLI and Python helper
check_scienceqa_answers.py               answer-only development check; audited train only
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
