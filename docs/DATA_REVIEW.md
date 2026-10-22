# Reviewing ScienceQA candidates

The metadata filter produces a candidate pool, not a certified diagram-dependent benchmark. Review candidates before looking at model outputs. Preserve original splits and use train/validation questions for prompt development only.

Review IDs come from the pinned Hugging Face release: `train_000042` is zero-based source row 42 of the training split, assigned before filtering. Images use the same IDs, for example `data/processed/scienceqa/images/train_000042.png`. These are project IDs, not the author's original question IDs. Record the dataset revision alongside completed reviews and regenerate/recheck selections if the source revision changes. Reviews or selections from the previous original-ID exports cannot be joined directly to these new IDs.

Copy `data/processed/scienceqa/visual_review_template.csv` to a separately named review file (for example, `manifests/scienceqa/visual_review_reviewer1.csv`) before editing. Do not overwrite a completed review with a preparation run. The template contains IDs and local image paths, with blank fields for the label, visual type, reviewer and rationale. Look up each question, context and ordered choices in `data/processed/scienceqa/questions/<split>.jsonl`, then view its image. References are unnecessary for this selection task.

For `visual_need`, choose exactly one:

| Label | Selection rule |
| --- | --- |
| `image-required` | Solving the specific question requires inspecting image-specific information: labels, arrows, positions, relationships, plotted values or pictured features not supplied by the text/context/choices alone. |
| `image-helpful` | The image makes the question clearer, but the text supplies enough information to solve it. |
| `image-not-required` | The image is decorative, redundant, or unrelated to the reasoning needed. |
| `uncertain` | Legibility, ambiguity, or possible text-only shortcuts prevent a confident classification. Record why. |

Imagine covering the image while keeping the entire public question input visible. Identify the specific missing fact, rather than judging whether a particular model might know the answer. Mention that missing visual fact in the rationale. Do not classify a question from model correctness or the reference explanation.

For `visual_type`, use `diagram`, `graph-or-table`, `photograph`, `mixed`, `other` or `uncertain`. A natural-science image can be a photograph, so the metadata filter does not certify diagram status. For a narrow science-diagram claim, the recommended main pool is `image-required` items with `diagram`, `graph-or-table`, or a justified diagram-containing `mixed` label. Keep photographs available for a separately described analysis. Whether graphs/tables are included in the final main pool remains a protocol choice.

Before final test inference, have two reviewers independently label test candidates without model outputs; reconcile disagreements and retain the original labels and rationale. Pilot the rules on development items first. Check the integrity audit's cross-split overlap groups and decide how to handle repeated items/images before sampling. Shared image alone need not mean duplicate question.

The next step is to review development candidates and choose approximately 10–20 varied examples for model feasibility. Final test sampling, random seed, overlap policy, sample size and final image-type scope are intentionally not implemented or frozen yet. Commit the agreed review results and final IDs when the protocol is settled; run every experimental condition on those same final IDs.
