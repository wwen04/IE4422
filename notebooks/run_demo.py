"""Run one development item using one model and the existing prototype stages."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_scienceqa_tutor import load_question, question_prompt
from tutor_pipeline import extract_evidence, diagnose, generate_hint
from notebooks.colab_support import complete_checkpoint, read_json, write_json


def run_demo(tutor, item, student_answer, explanation, levels, save):
    """Reuse the same model and the same raw intermediate strings for this item."""
    if not levels or any(level not in (1, 2, 3) for level in levels):
        raise ValueError("Request hint levels 1, 2 or 3.")
    prompt = question_prompt(item)
    result = {"item_id": item["item_id"], "split": item["split"], "hints": {}}
    result["evidence"] = extract_evidence(tutor, item["image_path"], prompt)
    save(result)
    result["diagnosis"] = diagnose(tutor, item["image_path"], prompt,
                                    student_answer, explanation, result["evidence"])
    save(result)
    for level in levels:
        result["hints"][str(level)] = generate_hint(tutor, item["image_path"], prompt,
                                                   result["evidence"], result["diagnosis"], level)
        save(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--levels", type=int, choices=(1, 2, 3), nargs="+", default=[1, 2, 3])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent", type=Path)
    args = parser.parse_args()
    attempt = read_json(args.attempt)
    if not attempt["answer"].strip() or not attempt["explanation"].strip():
        parser.error("Provide the student's actual answer and explanation before generation.")
    item = load_question(ROOT, args.split, args.item_id, data_source="final")
    config = read_json(ROOT / "notebooks/colab_config.json")
    model_folder = ROOT / "models" / config["model_revision"]
    if not complete_checkpoint(model_folder, config):
        parser.error("Run the model-download cell first; the local checkpoint is incomplete.")

    import torch
    from qwen_engine import IMAGE_SIZE, QwenTutor

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder = args.output_dir / run_id
    folder.mkdir(parents=True)
    versions = {name: metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "datasets", "huggingface-hub", "bitsandbytes")}
    record = {"run_id": run_id, "model": config, "packages": versions,
              "dataset": read_json(ROOT / "configs/scienceqa.json"),
              "audited_selection": read_json(ROOT / "data/final/scienceqa/selection.json"),
              "bundle": read_json(ROOT / "bundle_info.json"),
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
              "torch_build": torch.__version__,
              "quantization": "NF4 4-bit; double quantization; float16 compute", "image_size": IMAGE_SIZE,
              "generation": {"do_sample": False, "evidence_tokens": 320,
                             "diagnosis_tokens": 250, "hint_tokens": 160},
              "levels": args.levels, "student_attempt": attempt, "status": "loading_model"}
    persistent = args.persistent / run_id if args.persistent else None

    def save(result):
        write_json(folder / "result.json", result)
        write_json(folder / "run.json", record)
        if persistent:
            persistent.mkdir(parents=True, exist_ok=True)
            for name in ("result.json", "run.json"):
                shutil.copy2(folder / name, persistent / name)

    save({"item_id": item["item_id"], "split": item["split"], "hints": {}})
    start = time.perf_counter()
    tutor = QwenTutor(str(model_folder))
    torch.cuda.synchronize()
    record["model_load_seconds"] = time.perf_counter() - start
    record["status"] = "generating"
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = run_demo(tutor, item, attempt["answer"], attempt["explanation"], args.levels, save)
    torch.cuda.synchronize()
    record.update(status="complete", generation_seconds=time.perf_counter() - start,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated())
    save(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved:", folder)
    # Exiting this process releases the model. A new generation-cell run loads again.


if __name__ == "__main__":
    main()
