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

The first run downloads approximately 626 MB of Parquet files containing text and image bytes. Allow a few GB for the downloaded files, loading cache and exported candidates. No account/token is required for this public dataset. To reuse the pinned cache without network access:

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

Image presence alone does not establish that a question needs visual reasoning. Complete [the visual review procedure](docs/DATA_REVIEW.md) before final evaluation selection. No final sample is frozen and no model outputs have been used for selection.

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

This remains the original exploratory **evidence → tentative diagnosis → one requested hint** prototype. Evidence/diagnosis are unparsed model-produced strings, not verified annotations. Diagnosis conditions hints on a student's response, which is outside the planned response-independent experiment. This reorganisation does not implement the final methodology: evidence validation/caching, three-hint history, disclosure checking, A–E comparisons and human evaluation remain pending. Development uses train/validation; preserve test items for the agreed final evaluation.

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
run_tutor.py / inspect_diagram.py        optional custom-diagram entry points
requirements.txt / requirements-lock.txt shared environment and constraints
notebooks/                              Colab notebook, source ZIP and helpers
configs/ / tests/ / docs/               configuration, checks and review guidance
data/                                    ignored by Git
  cache/huggingface/                      pinned download and Hugging Face loading cache
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
