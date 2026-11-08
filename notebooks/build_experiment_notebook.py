"""Build the separate Colab batch-experiment notebook from reviewable cells."""

import json
from pathlib import Path
import textwrap


def build(destination=None):
    cells = []

    def cell(kind, source):
        item = {"cell_type": kind, "id": f"experiment-cell-{len(cells):02d}", "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            item.update(outputs=[], execution_count=None)
        cells.append(item)

    cell("markdown", """
    # IE4422: batch experiments A, B and D

    Use a fresh **GPU** runtime. This notebook runs experiments separately from the browser tutor.
    ScienceQA is selected by default and uses audited training questions; AI2D uses its frozen
    100-question test sample.
    Select one dataset at a time. You can run A, B and D sequentially in this session on that dataset.
    Each condition is a separate saved run. Do not launch simultaneous runs on the same GPU.
    Generation controls default to off. B and D each produce three independent hint tiers per question.
    D extracts evidence separately for every hint, without caching or learner attempts.
    D uses the same minimum-sufficient-observations prompt as the browser tutor. The public question,
    context and options guide relevance but are not visual evidence. Output is at most 10 concise
    numbered observations with types and locations under a 320-token limit; evidence is unverified.
    AI2D's same frozen 30 questions supply both B and D judging packages (90 planned hints each).
    ScienceQA reuses the exact 100 judging questions from the completed B run (300 planned hints per condition).
    No judge is called automatically. Keep datasets and condition results separate.
    """)
    cell("markdown", "## 1. Upload the experiment source ZIP\nUpload **IE4422_Experiments_Source.zip** to use the minimum-sufficient-observations evidence prompt. For this prompt update, keep the existing notebook and rerun this upload cell and the setup cells with the new source ZIP. This bundle is built separately from the UI source ZIP. Use a fresh runtime when replacing an older bundle.")
    cell("code", """
    from pathlib import Path, PurePosixPath
    from datetime import datetime, timezone
    from io import BytesIO
    import json
    import stat
    import subprocess
    import sys
    import zipfile
    from google.colab import files

    if not Path('/content').is_dir() or not sys.platform.startswith('linux'):
        raise RuntimeError('Open this notebook in Google Colab with a GPU runtime.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Select Runtime > Change runtime type > GPU, then reconnect.')
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise RuntimeError('Upload exactly one experiment source ZIP.')
    archive_name, archive_content = next(iter(uploaded.items()))
    if not archive_name.lower().endswith('.zip'):
        raise RuntimeError('Upload the experiment source ZIP.')
    SOURCE_ROOT = Path('/content/ie4422_experiment_sources') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    SOURCE_ROOT.mkdir(parents=True)
    with zipfile.ZipFile(BytesIO(archive_content)) as source:
        names = set()
        for info in source.infolist():
            member = PurePosixPath(info.filename)
            target = (SOURCE_ROOT / info.filename).resolve()
            if (member.is_absolute() or '..' in member.parts or '\\\\' in info.filename
                    or ':' in info.filename or info.filename in names
                    or stat.S_ISLNK(info.external_attr >> 16) or not target.is_relative_to(SOURCE_ROOT.resolve())):
                raise ValueError('Unsafe or duplicate ZIP member: ' + info.filename)
            names.add(info.filename)
        info = json.loads(source.read('IE4422/bundle_info.json'))
        if info.get('kind') != 'batch_experiments':
            raise ValueError('This is not the batch-experiment source bundle.')
        source.extractall(SOURCE_ROOT)
    WORK = SOURCE_ROOT / 'IE4422'
    sys.path.insert(0, str(WORK))
    for name in list(sys.modules):
        if name in ('notebooks', 'experiments') or name.startswith(('notebooks.', 'experiments.')):
            del sys.modules[name]
    from notebooks.colab_support import run_visible
    from notebooks.experiments_colab import prepare_dataset, run_folder, run_experiment, package_run, summarize_run, import_judgments
    print('Source:', WORK)
    """)
    cell("markdown", "## 2. Choose the dataset and storage\nScienceQA is selected for this run. Its full D experiment uses the 613 audited training questions (1,839 planned hint outputs and up to 3,678 model calls). Use a new run label and keep RESUME=False for the updated prompt. Each saved run belongs to one dataset and one condition. To switch datasets later, change DATASET, rerun this settings cell and Cell 4, then run the required experiment cells. The environment and downloaded model can be reused. Enable resume only for an existing run with the same dataset, label, source and checkpoint.")
    cell("code", """
    DATASET = 'scienceqa'  # @param ['scienceqa', 'ai2d']
    USE_DRIVE = True  # @param {type:'boolean'}
    SAVE_MODEL_TO_DRIVE = True  # @param {type:'boolean'}
    REBUILD_DATA = False  # @param {type:'boolean'}
    RUN_LABEL = ''  # @param {type:'string'}
    RESUME = False  # @param {type:'boolean'}
    PERSISTENT = None
    if USE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        PERSISTENT = Path('/content/drive/MyDrive/IE4422')
    if not RUN_LABEL:
        RUN_LABEL = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    RUNS = {condition: run_folder(WORK, DATASET, condition, RUN_LABEL, PERSISTENT) for condition in ('A', 'B', 'D')}
    print('Run label:', RUN_LABEL)
    for condition, folder in RUNS.items():
        print(condition, folder)
    """)
    cell("markdown", "## 3. Set up the project environment\nReuse Colab's CUDA wheels with the existing project dependency setup.")
    cell("code", """
    ENVIRONMENT = Path('/content/ie4422_experiment_env')
    run_visible([sys.executable, WORK / 'notebooks/colab_support.py', 'setup', WORK, ENVIRONMENT], cwd=WORK)
    PYTHON = ENVIRONMENT / 'bin/python'
    """)
    cell("markdown", "## 4. Prepare the dataset and validate the experiment code\nFor AI2D, preparation validates all 3,088 source rows and exports the frozen 100-question evaluation sample. Prepared final data can be restored from Drive. This step does not load a model or run a judge. Some tests deliberately print rejected-command messages followed by `ok`; the final `OK` confirms that test suite passed.")
    cell("code", """
    prepare_dataset(WORK, PYTHON, DATASET, PERSISTENT, REBUILD_DATA)
    for test in ('test_judge_preparation.py', 'test_batch_experiments.py', 'test_experiment_colab.py'):
        run_visible([PYTHON, '-m', 'unittest', 'discover', '-s', 'tests', '-p', test, '-v'], cwd=WORK)
    """)
    cell("markdown", "## 5. Download or restore the fixed Qwen checkpoint\nThe pinned checkpoint needs approximately 7.5 GB before quantization. The existing Drive model cache can be reused.")
    cell("code", """
    command = [PYTHON, WORK / 'notebooks/colab_support.py', 'model', WORK]
    if PERSISTENT and SAVE_MODEL_TO_DRIVE:
        command.extend(['--persistent', PERSISTENT / 'models'])
    run_visible(command, cwd=WORK)
    model_config = json.loads((WORK / 'notebooks/colab_config.json').read_text())
    MODEL = WORK / 'models' / model_config['model_revision']
    print('Checkpoint:', MODEL)
    """)
    for condition, description in (("A", "answer-only check"), ("B", "generic hints"), ("D", "evidence-conditioned hints")):
        instructions = f"## Run {condition}: {description}\nEnable this control when ready. Wait for the previous experiment to finish before running this cell. Each command loads one model, writes every outcome immediately, and exits to release GPU memory."
        if condition == "D":
            instructions += "\n\nD calls the UI's shared minimum-sufficient-observations evidence prompt, then the existing tier hint prompt. Evidence stays unverified. Set RUN_D=True to start the full selected dataset. For this updated prompt, use a new run label and RESUME=False; completed A/B results can be retained. Do not resume a D run generated with an earlier evidence prompt."
        cell("markdown", instructions)
        cell("code", f"""
        RUN_{condition} = False  # @param {{type:'boolean'}}
        if RUN_{condition}:
            run_experiment(WORK, PYTHON, MODEL, DATASET, '{condition}', RUN_LABEL, PERSISTENT, RESUME)
        else:
            print('{condition} skipped; no inference performed.')
        """)
    cell("markdown", "## Prepare the B and D judge packages\nAI2D uses the same saved 30-question selection for both conditions: 90 planned hints each. ScienceQA uses the exact saved 100 questions already judged in B: 300 planned hints each, including the original calibration exclusions. No new question draw is made for the default packages. Generation selections are unchanged. Empty/failed generations remain in the private map. This step creates attachments only; it does not call a judge.")
    cell("code", """
    PREPARE_B_PACKAGE = False  # @param {type:'boolean'}
    PREPARE_D_PACKAGE = False  # @param {type:'boolean'}
    PACKAGE_NAME = 'judge_package'  # @param {type:'string'}
    PACKAGES = {condition: RUNS[condition] / PACKAGE_NAME for condition in ('B', 'D')}
    for condition, enabled in (('B', PREPARE_B_PACKAGE), ('D', PREPARE_D_PACKAGE)):
        if enabled:
            package_run(WORK, PYTHON, RUNS[condition], PACKAGE_NAME)
    print('Give a judge only the shared judge prompt and one case folder at a time. Keep the private map and original run away from judging contexts.')
    """)
    cell("markdown", "## Download saved results or a judging package\nA package ZIP includes the coordinator's private map. Give each judge only its individual image and hints file plus the judge prompt.")
    cell("code", """
    DOWNLOAD = False  # @param {type:'boolean'}
    ARTIFACT = 'B judge package'  # @param ['A run', 'B run', 'D run', 'B judge package', 'D judge package']
    if DOWNLOAD:
        import shutil
        condition = ARTIFACT[0]
        target = PACKAGES[condition] if 'judge package' in ARTIFACT else RUNS[condition]
        if not target.is_dir():
            raise FileNotFoundError('Run/package does not exist yet: ' + str(target))
        download = shutil.make_archive('/content/' + DATASET + '_' + condition + '_' + RUN_LABEL + '_' + ('judge_package' if 'judge package' in ARTIFACT else 'run'), 'zip', root_dir=target)
        files.download(download)
    """)
    cell("markdown", "## Import returned judgments and compute a separate summary\nChoose B or D. Upload a ZIP containing only judgment JSON files. Existing valid conflicting judgments are preserved; incomplete judging is reported explicitly.")
    cell("code", """
    CONDITION = 'B'  # @param ['B', 'D']
    UPLOAD_JUDGMENTS = False  # @param {type:'boolean'}
    COMPUTE_SUMMARY = False  # @param {type:'boolean'}
    INCLUDE_A_BREAKDOWN = True  # @param {type:'boolean'}
    if UPLOAD_JUDGMENTS:
        returned = files.upload()
        if len(returned) != 1:
            raise ValueError('Upload exactly one judgment ZIP.')
        name, content = next(iter(returned.items()))
        upload_path = Path('/content') / ('returned_judgments_' + CONDITION + '.zip')
        upload_path.write_bytes(content)
        print('Imported:', import_judgments(upload_path, RUNS[CONDITION] / 'judgments'))
    if COMPUTE_SUMMARY:
        answers = RUNS['A'] if INCLUDE_A_BREAKDOWN and (RUNS['A'] / 'result.json').is_file() else None
        summary = summarize_run(WORK, PYTHON, RUNS[CONDITION], PACKAGES[CONDITION], answers)
        print((summary / 'README.md').read_text())
        files.download(str(summary / 'metrics.json'))
    """)
    cell("markdown", "AI2D is held out: keep prompts/settings fixed after development on ScienceQA. A accuracy measures answering; Part A rubric judgments measure hint quality. Do not filter the judging selection by A correctness or generation success. Missing AI2D reference explanations are not invented. Results stay separated by dataset, condition and run.")
    notebook = {"cells": cells, "metadata": {"colab": {"name": "IE4422_Experiments.ipynb"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
    path = Path(destination) if destination else Path(__file__).with_name("IE4422_Experiments.ipynb")
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    print(build())
