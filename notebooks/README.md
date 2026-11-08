# Run the browser tutor on Google Colab

For separate AI2D/ScienceQA batch experiments A, B and D, use **IE4422_Experiments.ipynb** with **IE4422_Experiments_Source.zip**. Its controls cover Drive persistence, resume, judge-package downloads and returned-score summaries. See [the experiment guide](../experiments/README.md). Rebuild that bundle with `python notebooks/build_experiment_bundle.py`. The instructions below refer only to the browser tutor.

Share these two files together:

- `IE4422_Colab.ipynb`
- `IE4422_Source.zip`

The notebook launches the same `ui_server.py` used locally. Questions, hints, and answer checks all happen in the browser UI.

## Start the tutor

1. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and upload `IE4422_Colab.ipynb`.
2. Choose **Runtime → Change runtime type → GPU**. A T4 is suitable when available; use Python 3.12+. Use a runtime eligible for web-UI interaction under the [Colab FAQ](https://research.google.com/colaboratory/faq.html).
3. Run cells **1–7** in order. Upload the matching source ZIP when prompted. Cell 3 controls Drive saving, model caching, and rebuilding data. There is no workflow selector.
4. Run cell **8** to start and embed the UI. Choose a train/validation question, ask for a hint, or submit an answer and reasoning. The model loads on the first request and stays loaded for later requests. Test questions are unavailable in this session.
5. Cell **9** shows server status and logs. Rerun cell 8 with `RESTART_UI=False` to reuse the server, or `True` to replace it.
6. Set `STOP_UI=True` and run cell **10** when finished to release GPU memory.
7. Set `DOWNLOAD_RESULTS=True` in cell **11** to download saved UI results as a ZIP. With Drive enabled, completed results also save immediately to **My Drive/IE4422/runs/ui/**.

**Run all** starts the UI and leaves it running. Stop and download are disabled by default. Keep the notebook open while using its embedded UI; the Colab proxy provides session access, not public hosting. No tunnel account is needed.

## Setup and saved files

The notebook creates one project environment using Colab's existing CUDA PyTorch wheels. It uses `--without-pip --system-site-packages` and Colab's pip with an explicit project-Python target, avoiding the earlier `ensurepip` failure. Dependencies install without changing the notebook kernel. Setup preserves torch/torchvision and their active dependency constraints, including bounded requirements such as `setuptools<82`; overrides are recorded in `tmp/colab-dependencies/runtime_pins.json`. Setup imports the model adapter in the project environment before any model is loaded. The UI launcher preserves the environment's Python entry point, including Colab's symlink, so the server sees the same installed packages.

- **Dataset:** approximately 626 MB on the first download. Preparation preserves official splits, separates public inputs from references, and applies the saved visual-need audit. A matching prepared-data archive can be restored from Drive. Finalization revalidates the audit on every setup.
- **Model:** approximately 7.5 GB of original checkpoint files. Quantization happens when loading the model. The revision is pinned in `colab_config.json`. Allow at least 8 GB of Drive space for optional model caching, plus data/results.
- **Working files:** `/content/IE4422` is temporary. `USE_DRIVE=True` retains prepared data and results; `SAVE_MODEL_TO_DRIVE` controls checkpoint caching. `REBUILD_DATA=True` repeats preparation.
- **Results:** local files are `outputs/ui_*.json`; Drive copies live under `runs/ui/`. Cell 11 also exports recovery files retained after a failed Drive copy. GPU memory and in-memory job status do not survive shutdown.

Hints before an attempt use evidence extraction followed by one requested-tier hint. The UI and Experiment D call the same `extract_evidence()` function: the public question, context and options identify relevance, and the model returns at most 10 concise numbered observations with types and locations under a shared 320-token limit. Options guide attention but are not visual evidence. The prompt asks for minimum sufficient observations and excludes inference, calculations, answering and hint generation; its output remains unverified. Answer checks compare the selected choice in Python and assess reasoning against the reference lecture/solution. A correct choice with supported reasoning stops with positive feedback; otherwise diagnosis and hint generation follow. Evidence is regenerated on each request; prior hints are not passed as conversation history.

To use the updated minimum-sufficient-observations prompt in an existing Colab UI session, keep the notebook and upload the new `IE4422_Source.zip` with `UPLOAD_NEW_SOURCE=True` in cell 2, then rerun cells 3–8. Source replacement stops the old UI server so the next launch loads the updated extractor. No notebook code changed for this update; a fresh runtime is recommended when replacing an older bundle.

**Show inspection report** reveals reference material, so leave it closed during student attempts. Model-generated evidence, diagnoses, and assessments remain unvalidated prototype outputs.

## Troubleshooting

- **UI blank or connection refused:** run cell 9, then rerun cell 8 to recreate the iframe. Allow Colab kernel access if prompted. A fresh runtime needs setup again; localhost on your laptop is not the Colab server.
- **Settings or source changed:** set `RESTART_UI=True` and rerun cell 8. Uploading source or rebuilding dependencies/data stops the old server first. Use a fresh runtime when replacing an older bundle.
- **First request is slow:** the checkpoint is loading into GPU memory. Check cell 9; later requests reuse the model.
- **`No module named 'qwen_vl_utils'` on the first request:** older UI launchers followed the project Python symlink to system Python, which could not see the installed project packages. Upload the corrected notebook/source pair in cell 2, rerun cell 4, then launch cell 8. Source replacement stops the old worker and discards its launcher object. This fix does not require rebuilding ScienceQA or downloading the checkpoint again.
- **Port occupied:** change `UI_PORT`, or stop the process that owns that port. The launcher does not kill unrelated processes.
- **Drive copy failed:** local results remain in `outputs/`. Fix Drive access/space or download them in cell 11. A job completes only after its enabled Drive copy succeeds.
- **No CUDA GPU:** select a GPU runtime and reconnect. Colab does not guarantee availability; TPU and CPU runtimes cannot run this model adapter.
- **Older setup fails in `ensurepip` or a setuptools constraint:** upload the latest notebook/source pair and rerun cells 1–4. Setup avoids bootstrapping pip and respects Colab's CUDA dependency constraints.
- **Changed preparation source or audit:** set `REBUILD_DATA=True` and rerun cell 6. Candidate-only archives are not accepted as finalized data.
- **Interrupted download or insufficient Drive space:** rerun cell 7 after resolving access/space. Incomplete checkpoints are not accepted for loading.

## Rebuild the source ZIP

From the repository root:

```powershell
.\.venv\Scripts\python notebooks/build_bundle.py
```

The bundle contains the UI, shared modules, preparation/finalization, inspector, saved audit decisions, and their tests. `colab_support.py` handles setup/storage; `colab_ui.py` manages the server process, readiness, and logs. Alternate tutor runners, notebook demos, and experiment launchers are excluded. Existing research files remain separately under `experiments/` in the repository. Data, weights, environments, secrets, and local project notes are excluded.

Local tests cover notebook structure and server lifecycle without loading a model. Actual Colab proxy/iframe behavior and GPU inference still require a real Colab run.

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Colab port helpers](https://github.com/googlecolab/colabtools/blob/main/google/colab/output/_util.py), [Hugging Face checkpoint downloads](https://huggingface.co/docs/huggingface_hub/guides/download).
