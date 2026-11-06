# Use the verified hint run for Part A judging

Verified: 100 questions, 300 unique question-tier records,
300 nonempty outputs, 0 empty outputs,
and 0 generation errors. Input/prompt/image fingerprints,
per-tier coverage, run counts and gold-reference joins match. This verifies data integrity, not hint quality.
This packaging step has not run a judge. The original `hints.jsonl` and `run.json` have not been changed.

## Selected judging sample

This package selects **30 questions x 3 tiers = 90 planned outcomes**.
It contains **90 nonempty hints** to judge, with
0 empty outputs and
0 generation errors retained in the private map.
Question IDs come from the committed AI2D 30-question judging selection, shared across B and D.

Keep `record_map_PRIVATE.jsonl` as the private mapping of this package's planned outcomes.
AI2D automatically reuses `manifests/ai2d/test_judge_selection.json` for later B/D packages; do not draw a new sample.
The selection does not depend on hint quality, judge scores or generation success. Keep failed
generation outcomes in the planned denominator; do not replace their questions with successful ones.
Original case IDs are retained and may have gaps. Only the selected cases are present.

1. Open **attachments/judge_prompt.md** and copy its full contents. It contains one set of judge
   instructions, the tier table, Part A criteria and JSON format. Setup and analysis notes stay in
   `experiments/judge_rubric.md` and are excluded from the judge prompt, along with Parts B-D.
2. Open a case folder under **attachments/cases/**. Give a fresh judging context the full prompt
   and **only the hints.jsonl and image in this folder**, using the same model and reasoning setting throughout.
   The hint file includes question, context, ordered choices, tier, hint and evaluator-only gold references.
   Attach the PNG as an image directly; an image path alone does not supply the image.
3. Save the returned **judgment_<record_id>.json**. Repeat with each remaining case in a fresh judging context.
   Each case has exactly one hint: there are **90 individual judge requests**, matching Part A.
   Folder IDs give the expected judgment filenames. The case order is shuffled.
   A coordinator can dispatch one case per fresh worker without prior chat history; do not give a
   judging worker sibling hints, previous judgments, or private maps. Save each result immediately
   and resume only missing or invalid files. Existing pilot judgments are not final sample results.
4. Bring back the saved judgment JSON files (a ZIP of them is convenient). We can validate/merge them
   into judgments.jsonl and join the original IDs using the private map, then compute rubric metrics.

## Keep these files away from the judge

Do not attach the original generation `hints.jsonl`, `run.json` or `record_map_PRIVATE.jsonl`.
They expose generator identity/prompts,
condition or run provenance. The case-specific hints.jsonl is a blinded export: its actual hint text,
question/context/choices and original image bytes are unchanged. Gold references are added only here
for judging and were not supplied during hint generation. No Experiment A correctness is shown.
No source images were resized or edited.

## Settings and access checks

Keep the same ChatGPT model/mode and rubric for every case, and record the displayed model name,
evaluation date and any settings you can observe. A prompt cannot set temperature: record it as
not controlled/not exposed unless your interface explicitly supplies that setting. Do not claim that
a manual ChatGPT run used temperature 0 or a fixed model snapshot without evidence of those settings.
If the judge cannot view an image, it should report the problem instead of inventing visual ratings.
If .jsonl is rejected by the upload UI, upload a copy renamed to hints.txt without changing its contents.
If file creation is unavailable, save the JSON-only response under its expected filename yourself.

## Dataset credit

AI2D: Kembhavi et al. (2016), *A Diagram Is Worth A Dozen Images*, ECCV. [Source](https://huggingface.co/datasets/lmms-lab-encoder/ai2d). [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The source supplies no lecture or worked solution; empty fields mean unavailable.
