# Local outputs

- `experiments/<dataset>/<A|B|D>/<run-id>/`: saved model outputs, run manifest, optional judge package, returned judgments and summaries.
- `inspection/`: disposable question inspection pages.

Use the shared runner, package builder and summary commands in `experiments/README.md`. Dataset and experiment results stay separate. Completed runs and returned judgments are retained. With Drive enabled, Colab saves runs under My Drive/IE4422/runs/experiments/<dataset>/<experiment>/<run-label>/. This directory is ignored by Git.
