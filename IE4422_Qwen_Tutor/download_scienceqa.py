"""Download ScienceQA test subset and export reproducible image-question pairs."""
import argparse
import json
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--split', choices=['train','validation','test'], default='test')
    parser.add_argument('--out', default='scienceqa_samples')
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('derek-thomas/ScienceQA', split=args.split)
    selected = []
    for index, row in enumerate(ds):
        if row['image'] is not None and row['grade'] in ('grade7','grade8') and row['subject'] == 'natural science':
            selected.append((index, row))
            if len(selected) >= args.count:
                break
    result = []
    for i, (original_index, row) in enumerate(selected, start=1):
        name = f'question_{i:03d}.png'
        row['image'].convert('RGB').save(output / name)
        result.append({
            'id': f'{args.split}-{original_index}',
            'image': str((output / name).resolve()),
            'question': row['question'],
            'choices': row['choices'],
            'answer': row['answer'],
            'grade': row['grade'],
            'subject': row['subject'],
            'topic': row['topic'],
            'category': row['category'],
            'skill': row['skill'],
            'solution': row['solution']
        })
    with (output / 'questions.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'Exported {len(result)} examples from {args.split} to {output.resolve()}')
    print('Review exported images: not all science images require visual interpretation.')

if __name__ == '__main__':
    main()
