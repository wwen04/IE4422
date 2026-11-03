# Run IE4422 on Google Colab

Send teammates these two files together:

- `IE4422_Colab.ipynb`
- `IE4422_Source.zip`

No local GPU, Python installation, GitHub access or Hugging Face account is needed for these public downloads.

1. Open [Google Colab](https://colab.research.google.com/) and choose **File → Upload notebook**. Upload `IE4422_Colab.ipynb`.
2. Choose **Runtime → Change runtime type → GPU** (a T4 is a suitable starting point if offered). Use a current Python 3.12+ runtime. A TPU does not work with this CUDA engine.
3. Run the notebook cells in order. When prompted, upload the current `IE4422_Source.zip` and approve mounting your own Google Drive. Leave `UPLOAD_NEW_SOURCE=True` when replacing an older bundle.
4. Leave `INTERFACE='answer checker'` for the new answer-only training check. The notebook creates one project environment, prepares ScienceQA, applies the bundled visual-need audit, and restores/downloads the pinned Qwen checkpoint.
5. Step 8 runs `check_scienceqa_answers.py` with one model instance and one direct call per question. `CHECK_LIMIT=10` checks the first ten audited training questions in saved order; set it to 0 for the full audited training set. The model returns one option letter without an explanation. The checker loads training reference letters only after generation. It does not train the model, select validation/test questions, or run hint stages.
6. Each batch retains raw responses, predictions, failures, timing and settings in a new `outputs/answer_checks/<run-id>/` folder. Scoring adds per-question correctness, accuracy and invalid/error counts; failed and invalid predictions remain in the denominator. Drive users receive a copy under **My Drive/IE4422/runs/answer_checks/<run-id>/**, including available partial files if the worker fails. Step 12 optionally downloads the latest run as a ZIP.
7. Run all performs the selected answer check and skips the browser/demo workflows. Rerun step 8 for a new batch. The worker releases its model when it exits. This is development feedback on the current audited Grade 7–8 training selection, not held-out accuracy or evidence of hint quality.

The checker normalizes whitespace and letter case, then accepts only a single supplied option letter. Extra text, empty responses and out-of-range letters are invalid. It records `predictions.jsonl` incrementally, a `run.json` status/settings file, and a scored `result.json`. Missing or invalid references stop scoring. The first-N sample is deterministic and should not be presented as a representative benchmark.

To use the existing interactive tutor, choose `INTERFACE='browser UI'` in step 3. Step 9 starts `ui_server.py` and embeds it through Colab's port proxy. Choose an audited train/validation question, ask for a hint or submit an answer and reasoning. The model loads on the first request and stays loaded for later requests. Test items are unavailable through this UI/API. UI results are saved locally as `outputs/ui_*.json` and, with Drive enabled, immediately copied to **My Drive/IE4422/runs/ui/**. Step 10 shows status/logs; rerun step 9 with `RESTART_UI=False` to reuse the server or True to replace it. Set `STOP_UI=True` in step 11 to release GPU memory. In browser mode, Run all leaves the server running and skips the checker/demo workflows. Step 12 exports saved UI results, including recovery files after a failed Drive copy.

Google's [Colab FAQ](https://research.google.com/colaboratory/faq.html) restricts interacting primarily through a web UI on free managed runtimes without a positive compute-unit balance. Use an eligible runtime for the browser UI. Embedding the app does not remove that restriction. Keep the notebook open; the proxy is session access, not public hosting. No tunnel service or tunnel account is needed. Google's separate-tab helper is deprecated because of browser security changes, so this notebook uses its supported iframe helper.

For the original workflow, choose `INTERFACE='notebook demo'` in step 3, then run setup and the three optional demo cells. It displays one audited question, asks for the student's actual answer/reasoning, and generates three hint levels. Its generation worker stops any UI server first and releases its own model when it exits. Demo runs retain their stage results, settings and package/GPU metadata under **My Drive/IE4422/runs/<run-id>/**.

The notebook is a launcher; all project jobs use one project environment with access to Colab's existing CUDA PyTorch. Creation uses `--without-pip --system-site-packages`, bypassing the failing `ensurepip` bootstrap in Colab's system Python. Colab's existing pip explicitly targets the project Python through `--python`, so installs go into the project environment without bootstrapping another pip. Project packages install without modifying the notebook kernel, so no kernel restart is needed. Setup checks this pip route before installing dependencies, and displays command output/errors live.

The profile preserves the actual torch/torchvision wheels and respects all their active dependency limits, including exact pins and ranges such as `setuptools<82`. Compatible shared pins are retained. If a Windows transitive pin conflicts, the profile uses a compatible installed runtime version or lets pip resolve the required range inside the project environment. The local lock is unchanged, and the profile records its overrides in `tmp/colab-dependencies/runtime_pins.json`. The relevant project dependency graph and CUDA availability are checked. Colab's unrelated preinstalled applications are not part of this environment's validation.

## Downloads and saved files

- Dataset: approximately 626 MB on the first fresh preparation. Public questions and evaluation references remain in separate files. Preparation retains all official splits; the answer checker reads only training exports. The existing tutor demos offer train/validation, and never select test items.
- Model: approximately 7.5 GB of original weights plus small supporting files. Four-bit quantization happens during model loading; it does not reduce the original download. The model ID and exact revision are in `colab_config.json`.
- Working files live in Colab's temporary `/content/IE4422` folder. Drive retains a prepared-data ZIP, an optional complete checkpoint folder, and generated results. Allow at least 8 GB free Drive space for model caching, plus space for data/results.
- On later sessions, the notebook restores matching artifacts from Drive. It checks preparation source/configuration, finalization code and saved audit decisions before reusing data. The archive includes processed and final exports; finalization validates the reviewed sources and rebuilds the final set before question selection. It checks the model revision and required files before loading a checkpoint.
- Set `REBUILD_DATA=True` to repeat full dataset preparation. Set `SAVE_MODEL_TO_DRIVE=False` to download weights to temporary Colab storage each new session. With `USE_DRIVE=False`, results can be downloaded at the end, but temporary files disappear when Colab deletes the runtime.
- UI server: one model is loaded lazily and reused across requests until stopped/restarted. This GPU-memory state cannot be saved to Drive. Disk checkpoint caching survives sessions; GPU memory and in-memory job status do not. UI files contain the generated result and submitted attempt; the optional demo additionally records runtime settings, package/GPU versions and per-stage partial results.

This demonstrates the current exploratory prototype. Pre-answer UI hints do not load reference answers. Submitted-answer checks compare the selected choice with the separate reference and may use the reference lecture/solution to assess reasoning. **Show inspection report** explicitly reveals reference material, so leave it closed during student attempts. The optional three-hint demo does not send references to the model. Model outputs remain unvalidated; research comparisons, validated evidence caching, hint history and disclosure checking remain pending.

## Troubleshooting

- **UI blank or refused to connect:** select browser UI in step 3, run step 10, check the server is running, and rerun step 9 to recreate its iframe. Keep the notebook open and allow Colab kernel access if prompted. A fresh runtime needs setup again. Localhost on your laptop is not the Colab server.
- **Server settings changed:** set `RESTART_UI=True`, rerun step 9, then reset it to False. Restart interrupts current jobs. Replacing source, dependencies or prepared data also stops the old worker before changing its files.
- **UI first request is slow:** the server is loading the cached checkpoint into GPU memory. Check step 10 for failures; later requests reuse that model. Do not also run a second model-loading worker.
- **Port already in use:** choose another `UI_PORT`, or stop the server that owns it. The launcher does not kill unrelated processes.
- **Drive result copy failed:** local results remain in `outputs/`. Fix Drive space/access or use step 12 to download recovery results. UI jobs are marked complete only after the enabled Drive copy succeeds; answer-check `run.json` describes the worker's local run, and the notebook reports any subsequent Drive-copy failure.

- **`ensurepip` failed in an older bundle:** download both updated files and open the updated notebook. Run cells 1–4, upload the updated ZIP in cell 2, and leave `UPLOAD_NEW_SOURCE=True`. The revised setup skips `ensurepip` and can reuse the partially created environment. Keep the existing Drive folder and saved files.
- **torch 2.11 requires `setuptools<82`, but a constraint requests 84.2.1:** download the latest source ZIP, rerun cell 2 with `UPLOAD_NEW_SOURCE=True`, then rerun cell 4. The corrected profile handles bounded requirements as well as exact pins. Keep the runtime and Drive files.
- **No CUDA GPU:** select a GPU runtime and reconnect. Colab does not guarantee GPU availability. A GPU in your laptop is unnecessary.
- **Old Python or PyTorch:** select a current Colab runtime. The notebook stops rather than changing the shared data stack or silently using CPU inference.
- **Installation or generation error:** keep the visible error. Check `run.json` and any completed `result.json` under Drive; a run is successful only when its status is `complete`. Re-run the failed cell after fixing the cause.
- **Insufficient Drive space:** turn off model caching or free space in your Drive, then rerun the model cell. A partially copied checkpoint is not reused as complete.
- **Interrupted model download:** rerun the model cell; Hugging Face can reuse completed files. Dataset caches are written only after full preparation.
- **Source files edited:** rebuild the ZIP so teammates run the updated code, not an older snapshot. Start a fresh Colab runtime when replacing an already uploaded source bundle.
- **Old prepared-data archive or changed audit:** set `REBUILD_DATA=True` and rerun cell 6 with the latest source ZIP. Candidate-only archives are not accepted as an audited dataset. Downloads can still reuse the local Hugging Face cache in the same runtime.
- **Question ID excluded by the audit:** choose an ID from `manifests/scienceqa/<split>_final_ids.json`, or leave `ITEM_ID` blank to select the first retained development item. The demo does not fall back to the candidate pool.

From the repository root:

```powershell
.\.venv\Scripts\python notebooks/build_bundle.py
```

`check_scienceqa_answers.py` at the bundle root provides the standalone training answer check. `colab_support.py` contains setup/storage helpers, `colab_ui.py` manages the UI process/readiness/logs, `run_demo.py` calls the existing model/prompt stages, and `build_bundle.py` packages an explicit list of source files. Model weights, datasets, environments, secrets and local proposal/agent notes are excluded from the source ZIP.

Local tests and notebook structure can be checked without a GPU. An actual Colab CUDA run is still required to verify assigned-hardware feasibility and model output quality.

Sources: [Colab storage/resource limits and web-UI restrictions](https://research.google.com/colaboratory/faq.html), [Colab port helpers](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py), [Hugging Face checkpoint downloads](https://huggingface.co/docs/huggingface_hub/guides/download).
