# Planned Colab entry point

Colab implementation is deferred. Local execution is documented in the root README and uses one shared environment from `requirements.txt` and `requirements-lock.txt`.

The future `colab_demo.ipynb` will orchestrate these same files rather than duplicate preparation logic or prompts:

1. Inspect the runtime, install compatible shared dependencies and mount Drive.
2. Obtain a fixed source version and work under a local repository directory.
3. Run `prepare_data.py`, or restore matching prepared outputs with the same repository-relative paths. Keep reference exports separate from public inputs.
4. Download or restore a pinned, complete Qwen checkpoint and copy it to local storage.
5. Create one `qwen_engine.QwenTutor` instance. Use `load_question` and `run_question` from `run_scienceqa_tutor.py` for subsequent calls with that instance.
6. Display development outputs and save completed results and settings to Drive.

Drive will retain prepared-data archives, checkpoint files and results. Local storage will hold working copies. The notebook must account for the preparation script's current local `HF_HOME` setting before offering a different cache location.

The current Python API supports model reuse; it does not yet cache/validate extracted evidence or implement the full experiment. Recheck the actual Colab Python/GPU/dependencies before claiming compatibility. Do not silently change precision, quantization or prompts between experimental conditions.
