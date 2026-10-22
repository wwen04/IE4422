"""Step 2: run the whole science tutor from a command line."""
import argparse
import json
from tutor_pipeline import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='images/science_diagram.png')
parser.add_argument('--question', required=True)
parser.add_argument('--answer', required=True)
parser.add_argument('--explanation', required=True)
parser.add_argument('--hint-level', type=int, choices=[1, 2, 3], default=1)
parser.add_argument('--model-path', default=None)
parser.add_argument('--output', default='tutor_result.json')
args = parser.parse_args()
result = run_pipeline(args.image, args.question, args.answer, args.explanation, args.hint_level, args.model_path)
print(json.dumps(result, indent=2, ensure_ascii=False))
with open(args.output, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f'\nSaved to {args.output}')
