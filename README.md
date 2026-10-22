# Visual evidence-guided science hints

Student research project comparing progressive hints with and without structured visual evidence, using the same frozen vision-language model. No training or fine-tuning.

## Qwen tutor integration

The ScienceQA preparation workflow below is retained. `IE4422_Qwen_Tutor/` adds the local Qwen2.5-VL-3B prototype: diagram reading, tentative misconception diagnosis and a targeted hint. `run_scienceqa_tutor.py` connects that prototype to the public JSONL questions exported by `prepare_data.py`, preserving the team's IDs, supplied context, option order and original images. It does not load reference files. Student answers and explanations must be provided explicitly.

See [Qwen setup and usage](docs/QWEN_INTEGRATION.md). This remains a prototype; it does not implement the complete comparison experiment or validate the quality of diagnoses/hints.

## Prepare ScienceQA

We use only Hugging Face's [`derek-thomas/ScienceQA`](https://huggingface.co/datasets/derek-thomas/ScienceQA). Its supplied train, validation and test splits are preserved. The dataset revision is pinned in `configs/scienceqa.json`, alongside the subject and grade filter. Importing downloads the release once and caches it locally; there is no separate metadata download or manual ZIP setup.

From the repository root, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-data.txt
.\.venv\Scripts\python prepare_data.py
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Use Python 3.12 or newer for the current dependency pins; Python 3.14.0 on Windows is the verified environment. On macOS/Linux, use `.venv/bin/python` instead. Future model-stack/GPU compatibility is still unverified. Install only `requirements-data.txt`: it automatically applies the version constraints in `requirements-data-lock.txt`, so keep both files in the repository but do not install them separately.

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

## Folders and GitHub teamwork

```text
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
