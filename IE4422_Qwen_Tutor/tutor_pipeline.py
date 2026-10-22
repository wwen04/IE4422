"""Evidence extraction -> tentative misconception diagnosis -> targeted hint."""
import json

# Illustrative categories only. Replace or refine from research-based inventories.
MISCONCEPTIONS = {
    "arrow_direction": "Misinterprets what a diagram arrow represents or its direction",
    "label_confusion": "Confuses the identity or meaning of labels",
    "scientific_relationship": "Misunderstands the scientific relationship depicted",
    "concept_application": "Recognizes components but incorrectly applies the underlying concept",
    "unrelated_visual_detail": "Focuses on irrelevant diagram details",
    "insufficient_evidence": "Student response does not support a confident diagnosis",
}


def extract_evidence(tutor, image, question):
    return tutor.ask(image, f"""Science question: {question}
Describe only visible diagram evidence relevant to this question.
Return a JSON object with keys: entities, labels, arrows, relationships, uncertainties.
Do not guess missing or unreadable details. Return JSON only.""", max_new_tokens=320)


def diagnose(tutor, image, question, student_answer, explanation, evidence):
    categories = json.dumps(MISCONCEPTIONS, indent=2)
    return tutor.ask(image, f"""Science question: {question}
Visual evidence (model-produced; may contain errors): {evidence}
Student answer: {student_answer}
Student explanation: {explanation}
Possible misconception categories: {categories}
Classify a likely misunderstanding ONLY if it is supported by what the student wrote.
Otherwise choose insufficient_evidence. Explain briefly, express uncertainty.
Do not reveal the question's correct answer.
Return a JSON object with keys: category, justification, confidence (low/medium/high).""", max_new_tokens=250)


def generate_hint(tutor, image, question, evidence, diagnosis, level=1):
    hints = {1: "Direct attention to relevant diagram evidence", 2: "Guide a useful scientific concept", 3: "Suggest a reasoning step without giving the result"}
    if level not in hints:
        raise ValueError("level must be 1, 2, or 3")
    return tutor.ask(image, f"""Science question: {question}
Visual evidence (unverified): {evidence}
Tentative diagnosis (unverified): {diagnosis}
Generate ONE concise, supportive tutoring hint at level {level}: {hints[level]}.
If the diagnosis is uncertain, ask one clarifying question instead of assuming a mistake.
Only reference diagram details you can verify visually. Do not state the answer, option letter, or reveal the solution.
Output only the hint.""", max_new_tokens=160)


def run_pipeline(image, question, student_answer, explanation, hint_level=1, model_path=None):
    # Load GPU dependencies only when inference is requested. Support both the
    # original scripts and imports from the repository-level ScienceQA runner.
    if __package__:
        from .qwen_engine import QwenTutor
    else:
        from qwen_engine import QwenTutor

    tutor = QwenTutor(model_path) if model_path else QwenTutor()
    evidence = extract_evidence(tutor, image, question)
    diagnosis = diagnose(tutor, image, question, student_answer, explanation, evidence)
    hint = generate_hint(tutor, image, question, evidence, diagnosis, hint_level)
    return {"evidence": evidence, "diagnosis": diagnosis, "hint": hint}
