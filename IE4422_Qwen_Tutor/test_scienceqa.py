"""Run existing QwenTutor engine on exported ScienceQA images (no reference-answer leakage)."""
import argparse
import json
from pathlib import Path
import sys

from qwen_engine import QwenTutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='scienceqa_samples/questions.json')
    parser.add_argument('--output', default='scienceqa_samples/qwen_results.json')
    parser.add_argument('--model-path', default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()

    questions = json.loads(Path(args.input).read_text(encoding='utf-8'))[:args.limit]
    if not questions:
        sys.exit('No examples found. First run download_scienceqa.py')
    print('Loading Qwen once...')
    tutor = QwenTutor(model_path=args.model_path)
    results = []
    for i, item in enumerate(questions, 1):
        options = '\n'.join(f'{chr(65 + j)}. {v}' for j, v in enumerate(item['choices']))
        instruction = (
            'You are assisting a student with a science diagram.\n'
            'Question: ' + item['question'] + '\nOptions:\n' + options + '\n\n'
            'Identify relevant visible features, including labels, arrows, and relationships. '
            'Then give ONE short hint that helps the student reason. '
            'Do not provide the correct answer, option letter, or full solution. '
            'If the image is ambiguous or not relevant, say so.\n'
            'Format: Visual evidence: ...\nHint: ...'
        )
        try:
            response = tutor.ask(item['image'], instruction, max_new_tokens=220)
            error = None
        except Exception as exc:
            response = ''
            error = f'{type(exc).__name__}: {exc}'
        # The reference answer is saved for offline inspection only and was NEVER included in the prompt.
        results.append({
            'id': item['id'],
            'question': item['question'],
            'image': item['image'],
            'output': response,
            'error': error,
            'reference_answer_index': item['answer'],
            'reference_answer_text': item['choices'][item['answer']]
        })
        print(f'[{i}/{len(questions)}] {item["id"]}\n{response or error}\n')
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', Path(args.output).resolve())

if __name__ == '__main__':
    main()
