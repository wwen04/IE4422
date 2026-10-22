"""Step 1: verify a single diagram can be read by Qwen."""
import argparse
from qwen_engine import QwenTutor

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='images/science_diagram.png')
parser.add_argument('--model-path', default=None, help='Optional local model folder instead of Hugging Face ID')
args = parser.parse_args()
model = QwenTutor(args.model_path) if args.model_path else QwenTutor()
response = model.ask(args.image, 'Identify the scientific diagram labels, objects, arrows, and relationships. Only report what is visible.')
print('\nMODEL RESPONSE:\n', response)
