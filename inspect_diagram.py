"""Inspect visible labels, objects and arrows in a user-supplied diagram."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--model-path', default=None,
                        help='Optional local model folder instead of Hugging Face ID')
    args = parser.parse_args(argv)
    if not Path(args.image).expanduser().is_file():
        parser.error(f'Missing science diagram: {args.image}')

    from qwen_engine import QwenTutor

    tutor = QwenTutor(args.model_path) if args.model_path else QwenTutor()
    response = tutor.ask(args.image, 'Identify the scientific diagram labels, objects, arrows, and relationships. Only report what is visible.')
    print('\nMODEL RESPONSE:\n', response)


if __name__ == '__main__':
    main()
