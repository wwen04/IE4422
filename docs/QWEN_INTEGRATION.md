# Qwen tutor integration

The team data-preparation code, pinned source revision, filters, manifests and audit reports are retained. The local prototype lives in `IE4422_Qwen_Tutor/`. The repository-level runner joins the workflows without using the prototype's independent dataset downloader.

## Prepare data

Follow the repository README with the team's data environment. `prepare_data.py` creates `data/processed/scienceqa/questions/<split>.jsonl` and the image files. Keep its dependency files unchanged. Downloaded data stays local.

## Set up Qwen separately

Use a separate environment because the team data dependency lock and the prototype's Transformers stack have not been verified together. Both environments share the exported files on disk. The local prototype environment used Python 3.12, PyTorch 2.11.0+cu128, Transformers 4.57.6 and Pillow 13.0.0; this is a record of installed versions, not a new dependency lock or a claim that GPU inference was verified during integration.

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv-qwen
.\.venv-qwen\Scripts\python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch using the [official selector](https://pytorch.org/get-started/locally/), using `.venv-qwen`'s Python, then install the original prototype requirements:

```powershell
.\.venv-qwen\Scripts\python -m pip install -r IE4422_Qwen_Tutor/requirements.txt
.\.venv-qwen\Scripts\python -c "import torch; print(torch.cuda.is_available())"
```

The model uses 4-bit quantization and requires an NVIDIA CUDA GPU. The first inference run downloads the model weights unless `--model-path` points to an existing model directory.

## Run a prepared question

Choose an ID from the prepared validation questions or `manifests/scienceqa/validation_valid_ids.json`. Read its question/options and provide an actual student response and explanation. For example, replace the placeholder explanation below:

```powershell
.\.venv-qwen\Scripts\python run_scienceqa_tutor.py --split validation --item-id validation_000003 --answer "A" --explanation "Replace with the student's actual reasoning" --hint-level 1
```

The default output is `outputs/scienceqa_tutor.json`. Use `--output` for another path. Output contains the item ID, split, visual evidence, tentative diagnosis and hint. Context and choices come from the public question; gold answers, lectures and solutions are not loaded. Evidence and diagnosis are model-generated strings, not verified structured annotations.

Use validation for development and preserve the test split for the agreed final evaluation. The runner does not enforce that research policy. Human visual review and final sample selection are still pending, as described in the team README.

## Original prototype scripts

From `IE4422_Qwen_Tutor/`, the original `test_qwen.py` and `run_tutor.py` scripts still accept a user-supplied diagram. See the README in that folder for their commands.

`download_scienceqa.py` and `test_scienceqa.py` are retained as legacy exploratory utilities. They use an independent, unpinned dataset download, different IDs and combined question/reference records. Use `prepare_data.py` and `run_scienceqa_tutor.py` for the shared project workflow. The original downloaded `scienceqa_samples/` and generated results are local-only and are not bundled here. No model weights or virtual environment are included.

## Verification

```powershell
python -m unittest discover -s tests -v
python run_scienceqa_tutor.py --help
```

The unit tests cover data preparation and the tutor connection using synthetic records and a fake tutor, including context/choice preservation, reference-field exclusion, missing inputs and output IDs. They require Pillow but no CUDA model download. They do not measure model accuracy or verify end-to-end GPU inference.
