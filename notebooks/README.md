# Run IE4422 on Google Colab

Send teammates these two files together:

- `IE4422_Colab.ipynb`
- `IE4422_Source.zip`

No local GPU, Python installation, GitHub access or Hugging Face account is needed for these public downloads.

1. Open [Google Colab](https://colab.research.google.com/) and choose **File → Upload notebook**. Upload `IE4422_Colab.ipynb`.
2. Choose **Runtime → Change runtime type → GPU** (a T4 is a suitable starting point if offered). Use a current Python 3.12+ runtime. A TPU does not work with this CUDA engine.
3. Run the notebook cells in order. When prompted, upload the current `IE4422_Source.zip` and approve mounting your own Google Drive. Leave `UPLOAD_NEW_SOURCE=True` when replacing an older bundle.
4. Leave the default settings for a first run. The notebook creates one project environment, prepares ScienceQA, displays a validation question, and asks for the student's actual response and reasoning.
5. It downloads the pinned Qwen checkpoint, then runs the existing prototype's evidence, diagnosis and three hint levels with one model instance. Completed stages/results are saved under **My Drive/IE4422/runs/**.

The notebook is a launcher; all project jobs use one project environment with access to Colab's existing CUDA PyTorch. Creation uses `--without-pip --system-site-packages`, bypassing the failing `ensurepip` bootstrap in Colab's system Python. Colab's existing pip explicitly targets the project Python through `--python`, so installs go into the project environment without bootstrapping another pip. Project packages install without modifying the notebook kernel, so no kernel restart is needed. Setup checks this pip route before installing dependencies, and displays command output/errors live.

The profile preserves the actual torch/torchvision wheels and respects all their active dependency limits, including exact pins and ranges such as `setuptools<82`. Compatible shared pins are retained. If a Windows transitive pin conflicts, the profile uses a compatible installed runtime version or lets pip resolve the required range inside the project environment. The local lock is unchanged, and the profile records its overrides in `tmp/colab-dependencies/runtime_pins.json`. The relevant project dependency graph and CUDA availability are checked. Colab's unrelated preinstalled applications are not part of this environment's validation.

## Downloads and saved files

- Dataset: approximately 626 MB on the first fresh preparation. Public questions and evaluation references remain in separate files. Test items are never selected for this demo.
- Model: approximately 7.5 GB of original weights plus small supporting files. Four-bit quantization happens during model loading; it does not reduce the original download. The model ID and exact revision are in `colab_config.json`.
- Working files live in Colab's temporary `/content/IE4422` folder. Drive retains a prepared-data ZIP, an optional complete checkpoint folder, and generated results. Allow at least 8 GB free Drive space for model caching, plus space for data/results.
- On later sessions, the notebook restores matching artifacts from Drive. It checks preparation source/configuration before reusing data and checks the model revision and required files before loading a checkpoint.
- Set `REBUILD_DATA=True` to repeat full dataset preparation. Set `SAVE_MODEL_TO_DRIVE=False` to download weights to temporary Colab storage each new session. With `USE_DRIVE=False`, results can be downloaded at the end, but temporary files disappear when Colab deletes the runtime.
- Notebook runtime settings and actual package/GPU versions are recorded with every run. Model loading happens once per generation-cell invocation, with evidence/diagnosis reused for that item's requested hint levels. The worker exits afterwards and releases its model memory. Rerunning generation starts a new model instance and keeps the previous run in a separate directory.

This demonstrates the current exploratory prototype. Its diagnosis uses a student response; evidence/diagnosis are unvalidated strings. Reusing these strings in one demo run is not a validated evidence cache or a frozen A–E protocol. Main-project research comparisons, hint history and disclosure checking remain pending. No reference answers/lectures/solutions are sent to the model.

## Troubleshooting

- **`ensurepip` failed in an older bundle:** download both updated files and open the updated notebook. Run cells 1–4, upload the updated ZIP in cell 2, and leave `UPLOAD_NEW_SOURCE=True`. The revised setup skips `ensurepip` and can reuse the partially created environment. Keep the existing Drive folder and saved files.
- **torch 2.11 requires `setuptools<82`, but a constraint requests 84.2.1:** download the latest source ZIP, rerun cell 2 with `UPLOAD_NEW_SOURCE=True`, then rerun cell 4. The corrected profile handles bounded requirements as well as exact pins. Keep the runtime and Drive files.
- **No CUDA GPU:** select a GPU runtime and reconnect. Colab does not guarantee GPU availability. A GPU in your laptop is unnecessary.
- **Old Python or PyTorch:** select a current Colab runtime. The notebook stops rather than changing the shared data stack or silently using CPU inference.
- **Installation or generation error:** keep the visible error. Check `run.json` and any completed `result.json` under Drive; a run is successful only when its status is `complete`. Re-run the failed cell after fixing the cause.
- **Insufficient Drive space:** turn off model caching or free space in your Drive, then rerun the model cell. A partially copied checkpoint is not reused as complete.
- **Interrupted model download:** rerun the model cell; Hugging Face can reuse completed files. Dataset caches are written only after full preparation.
- **Source files edited:** rebuild the ZIP so teammates run the updated code, not an older snapshot. Start a fresh Colab runtime when replacing an already uploaded source bundle.

From the repository root:

```powershell
.\.venv\Scripts\python notebooks/build_bundle.py
```

`colab_support.py` contains setup/storage helpers, `run_demo.py` calls the existing model/prompt stages, and `build_bundle.py` packages an explicit list of source files. Model weights, datasets, environments, secrets and local proposal/agent notes are excluded from the source ZIP.

Local tests and notebook structure can be checked without a GPU. An actual Colab CUDA run is still required to verify assigned-hardware feasibility and model output quality.

Sources: [Colab storage and resource limits](https://research.google.com/colaboratory/faq.html), [Hugging Face checkpoint downloads](https://huggingface.co/docs/huggingface_hub/guides/download).
