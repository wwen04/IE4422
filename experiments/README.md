# Batch experiments

The batch workflows run independently of the browser tutor. AI2D and ScienceQA share prompts and the model adapter, but retain separate selections, run folders, judging samples and results. No student attempts or reference-derived assessment results enter generation.

## Colab

For the minimum-sufficient-observations prompt update, keep the existing `notebooks/IE4422_Experiments.ipynb` and upload the new `notebooks/IE4422_Experiments_Source.zip` in Cell 1, then rerun setup Cells 2–5. No notebook code changed for this update; the rebuilt notebook refreshes instructions only. A fresh GPU runtime is recommended when replacing an older bundle. Start D with a new run label and `RESUME=False`; do not resume runs generated with an earlier evidence prompt. Completed A/B results can be retained. This is a separate notebook/bundle from the browser tutor. Select AI2D (default) or ScienceQA; then enable A, B and D individually. The notebook supports the pinned checkpoint, Drive persistence, interrupted-run resume, package downloads, returned judgment imports and summaries. Generation and judging-package controls default to off. No LLM judge is called by any script here.

One saved run belongs to one dataset and one condition. A single notebook session can run A, then B, then D sequentially on the same prepared selection. Do not start concurrent experiments on the same GPU: each runner loads its own model. To switch datasets later, change `DATASET`, rerun the settings cell and dataset-preparation cell, and then run the desired conditions. The environment and downloaded checkpoint can be reused, while dataset-specific outputs stay separate. Cell 4's software tests deliberately exercise rejected commands; a message ending in `ok` is an expected rejection test. Notebook consistency checks compare parsed JSON so Windows/Linux file line endings do not cause false failures.

Rebuild the experiment bundle after editing source:

```powershell
.\.venv\Scripts\python.exe notebooks/build_experiment_notebook.py
.\.venv\Scripts\python.exe notebooks/build_experiment_bundle.py
```

With Drive enabled, runs write each outcome directly to `My Drive/IE4422/runs/experiments/<dataset>/<A|B|D>/<run-label>/`. Prepared final exports use a fingerprinted archive under `prepared/`; the existing pinned checkpoint cache under `models/` can be shared with the UI notebook. Use the same run label across conditions. Resume requires the original source, inputs, prompts, checkpoint and image settings. Errors/empty outputs are retained, not regenerated to improve quality.

## AI2D

Prepare the source and freeze the judging subset before generation:

```powershell
.\.venv\Scripts\python.exe prepare_ai2d.py --offline
.\.venv\Scripts\python.exe -m experiments.prepare_judging --freeze-ai2d-selection
.\.venv\Scripts\python.exe -m experiments.run --dataset ai2d --experiment A --preflight
.\.venv\Scripts\python.exe -m experiments.run --dataset ai2d --experiment B --preflight
.\.venv\Scripts\python.exe -m experiments.run --dataset ai2d --experiment D --preflight
```

Omit `--offline` on the first download. Preflight loads public inputs and fingerprints only; it never imports a GPU model or opens reference exports. AI2D remains its original **test** split. The default is the frozen **100 questions**, never only those answered correctly by A. The committed `manifests/ai2d/test_judge_selection.json` records a fixed **30-question subset** of those 100 questions. Package preparation includes all three tiers for each of these 30 questions: **90 planned hints per condition**. The same IDs are reused for B and D, independently of outputs and answer correctness. The remaining 70 questions stay in the final dataset and still receive A predictions and B/D hints; only the judging workload is reduced.

To generate with the chosen checkpoint, omit `--preflight` and supply `--model-path` if needed:

```powershell
.\.venv\Scripts\python.exe -m experiments.run --dataset ai2d --experiment A --model-path <checkpoint-folder>
.\.venv\Scripts\python.exe -m experiments.run --dataset ai2d --experiment B --model-path <checkpoint-folder>
.\.venv\Scripts\python.exe -m experiments.run --dataset ai2d --experiment D --model-path <checkpoint-folder>
```

`--limit N` or repeated `--item-id` can create a pilot. A pilot that does not cover the complete frozen judging selection cannot produce the main 30-question package. For an interrupted run, supply `--resume --output-dir <existing-run-folder>` without new selection flags. Saved failures are retained; only missing records are generated. A trailing interrupted JSON fragment is quarantined during resume, not silently discarded as a valid outcome.

A follows one normal path: generate predictions, validate/load reference answers, calculate accuracy, write `result.json`, and mark the run complete. AI2D reference fingerprints accept LF/CRLF line-ending equivalents of the preparation report's checksum; changes to answer contents still fail validation. For a fresh run, use a new run label and `RESUME=False`; old run folders are preserved.

| Condition | Outputs for 100 questions | Model calls | Reference access |
|---|---:|---:|---|
| A: answer only | 100 answers | 100 | Answer key loaded only after generation for accuracy |
| B: generic hints | 300 hints | 300 | None |
| D: evidence-conditioned hints | 300 planned hint outcomes | 600 if both stages complete | None |

D calls the UI's shared `extract_evidence()` function in `tutor_pipeline.py`, with the same evidence prompt and shared 320-token limit, followed by the shared tier hint prompt (160-token limit). Prompt version `d-ui-prompt-with-evidence-v2` requests at most 10 brief numbered observations with types and locations, covering text, counts, objects, arrows, tables, colour legends and graphs. It explicitly excludes inference and answer-choice evaluation, while preserving letters that are visible image labels. Evidence remains raw text without format or correctness validation or repair. Evidence is regenerated independently for each question-tier pair; there is no cross-tier caching, diagnosis or student attempt. Both prompts, unedited evidence, unedited hint, failure status and timing are saved. Total generation timing includes both stages. Empty/failed evidence leaves a planned failed outcome; it does not trigger quality retries or a generic replacement hint. Model-produced evidence remains unverified, and no structured-evidence correctness claim is made merely because generation completed. This is the batch version of the current no-attempt two-stage method, not an independent evidence verifier.

For the first D run with this prompt, use `RESUME=False`; the completed B results can be reused. If a D run already exists with the older JSON evidence prompt, use a new D run label. Resume deliberately rejects changed prompts rather than mixing evidence formats in one run.

## Prepare and judge hints

After a B or D run completes:

```powershell
.\.venv\Scripts\python.exe -m experiments.prepare_judging --run-dir <completed-run-folder>
```

AI2D always uses its frozen 30 questions: **90 planned outcomes per condition**, **180 for B and D together**. ScienceQA's default package now automatically uses `manifests/scienceqa/train_judge_selection.json`: the **exact 100 questions already judged in B**, with all three tiers (**300 planned outcomes per condition**). It does not draw another sample. Failed/empty generations remain in the private map and are not replaced. Only nonempty hints have judge attachment cases. Existing packages are never overwritten; supply a new `--output-dir` if needed. ScienceQA's explicit `--sample-from`, `--sample-questions` for a different-sized pilot and `--all` options remain available; use the default fixed sample for the main B/D comparison.

The ScienceQA manifest was imported from the completed B private map, preserving the original input/image fingerprints and ten calibration exclusions. Its 100 IDs are reproducible using seed 42 on the sorted **603-question** pool after exclusions; seed 42 on all 613 questions selects different IDs. The saved IDs, rather than a fresh random draw, define the comparison. Existing B hints and its 300 returned judgments remain unchanged and can be reused. D's generation defaults still cover all 613 audited training questions; the default package extracts the same 100 questions as B. Generation selection and the separate 100-question judging selection are not the same thing.

Each `judge_package/` contains:

```text
README.md
record_map_PRIVATE.jsonl          coordinator-only mapping, including failed outcomes
attachments/
  judge_prompt.md                 shared rubric Part A instructions
  cases/r_000123/
    hints.jsonl                  exactly one unedited hint + question + tier + gold reference
    image_r_000123.png            original supplied image bytes
```

Give each fresh judge context only `judge_prompt.md` and **one case's** hint file and image. Never provide the private map, model/run metadata, sibling hints or D's evidence block. Use the same judge settings/rubric across conditions. AI2D lecture/solution fields are empty because the source does not supply them; judges use the diagram, question, choices, answer key and school-level science knowledge. Missing reference explanations are not invented. Unresolvable image/reference conflicts require review rather than fabricated ratings. **Rubric Part A** scores hints and is distinct from **Experiment A** answer accuracy.

Store returned `judgment_r_000123.json` files under that run's `judgments/`. They can be uploaded as a ZIP through the experiment notebook. Valid conflicting prior judgments are preserved instead of overwritten.

## Compute results

```powershell
.\.venv\Scripts\python.exe -m experiments.judge_validate <judgments-folder>
.\.venv\Scripts\python.exe -m experiments.summarize --run-dir <hint-run-folder> --output-dir <new-summary-folder>
```

Add `--answers-run-dir <matching-A-run-folder>` to break down hint scores by A correctness without filtering the judging sample. This join requires the same dataset, public-input fingerprints, checkpoint and image settings. Summaries include the good-hint composite, every component, disclosure distribution and mean actionability overall/per tier. Coverage and missing/invalid judgments are explicit. Conditional metrics use valid judgments only; the good-hint rate over all planned outcomes is available only once all nonempty hints have valid judgments, with failed/empty generations retained in its denominator. No confidence intervals or pooled cross-dataset score are invented. Compare B and D within each dataset; keep their same paired question/tier selections.

## Files and saved runs

There is one runner for both datasets and all three conditions. There are no duplicate dataset-specific launchers or old command aliases.

```text
experiments/
  run.py                         run A, B or D using --dataset and --experiment
  data.py                        load prepared data and fixed dataset/judging selections
  common.py                      shared prompts, answer scoring, hashes and metadata
  prepare_judging.py              verify B/D runs and export blinded cases
  judge_validate.py              validate returned judgment JSON
  summarize.py                   join judgments and calculate metrics
  legacy.py                      read the completed ScienceQA B record format
  judge_rubric.md                 evaluation instructions, separate from generation
  README.md
outputs/
  experiments/<dataset>/<A|B|D>/<run-id>/
  inspection/
```

`legacy.py` is a read-only format reader, not another experiment runner. It is retained because the completed ScienceQA B run has an older record layout. The shared package builder and summarizer recognize that format automatically. Existing saved runs, original packages and returned judgments are unchanged. Original command names such as `check_scienceqa_answers`, `generate_scienceqa_hints` and `prepare_hint_judging` were removed; use the shared commands above for both datasets. The obsolete output organizer, old audit-work archive and migration log were removed.

The original ScienceQA B run remains at `outputs/experiments/scienceqa/B/20261103T084643_855407Z/`. Its historical README may mention old commands; current instructions are in this guide.

Software tests use fake responses in temporary folders. No model generation or new judging is performed by software checks.
