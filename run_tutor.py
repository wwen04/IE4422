"""Run the tutor prototype on a user-supplied diagram and question."""
import argparse
import json
from pathlib import Path

from tutor_pipeline import run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--answer', required=True, help="Student response")
    parser.add_argument('--explanation', required=True, help="Student reasoning")
    parser.add_argument('--hint-level', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / 'outputs/custom_tutor.json')
    args = parser.parse_args(argv)
    if not Path(args.image).expanduser().is_file():
        parser.error(f'Missing science diagram: {args.image}')

    from qwen_engine import QwenTutor

    tutor = QwenTutor(args.model_path) if args.model_path else QwenTutor()
    result = run_pipeline(tutor, args.image, args.question, args.answer,
                          args.explanation, args.hint_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n',
                           encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'\nSaved to {args.output.resolve()}')


if __name__ == '__main__':
    main()
