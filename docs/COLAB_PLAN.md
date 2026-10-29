# Colab workflow status

The implementation and teammate instructions now live in [notebooks/README.md](../notebooks/README.md). Share `IE4422_Colab.ipynb` and `IE4422_Source.zip` together. No local GPU is required.

The notebook launches one project environment for both preparation and inference. It inherits Colab's CUDA PyTorch, retains the actual torch/torchvision versions, and reconciles every active direct runtime dependency requirement (exact pins and bounded ranges) against the Windows lock. Compatible shared pins remain; conflicting pins use a compatible installed runtime version or the required version range for pip to resolve inside the project environment. This covers the reported torch 2.11/setuptools<82 conflict. The notebook kernel and shared local requirements remain unchanged. The generated runtime profile records overrides, and actual main package/GPU versions are saved with each generated run.

Environment creation uses `--without-pip --system-site-packages` to bypass the user-reported Colab/Python-3.13 `ensurepip` failure. Colab's existing pip is checked and explicitly targets the project interpreter with `--python`. Child stdout/stderr are streamed through notebook output, and failed commands still stop the cell. The source-upload cell defaults to refreshing the ZIP and reloads the storage helper, preventing the older setup from remaining active after a bundle update.

Code, preparation and model loading run from `/content/IE4422`. Drive holds the prepared-data archive, optional fixed-revision model checkpoint and completed outputs. No change to the preparation script's local cache path is needed: preparation is a separate process in the local working copy. Cache reuse checks source/configuration and required files. Results are saved after completed stages, with incomplete runs retaining partial outputs.

`notebooks/run_demo.py` calls the existing extraction, diagnosis and hint functions with one model instance. It reuses raw evidence/diagnosis for the selected item's requested hint levels, then exits to release model memory. This is a demonstration policy, not a frozen experimental protocol or a validated evidence cache. The original response-conditioned diagnosis and missing history/disclosure checking remain methodology gaps.

The source ZIP contains an explicit source-only file list and a build record. It excludes data, weights, environments, secrets and local proposal/agent notes. Rebuild it with `python notebooks/build_bundle.py` whenever source or notebook content changes.

Local verification covers notebook structure/code, source packaging, safe archive extraction, prepared-data round trips, incomplete-checkpoint rejection, runtime-specific dependency pins and model/intermediate reuse with a fake tutor. Full CUDA inference in an actual Colab runtime remains unverified.
