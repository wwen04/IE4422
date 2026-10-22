
"""Load one Qwen2.5-VL model and reuse it for all tutor stages."""

from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


class QwenTutor:
    def __init__(self, model_path=MODEL_ID):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU was not detected. "
                "Install CUDA-enabled PyTorch and check nvidia-smi."
            )

        print("Loading Qwen2.5-VL-3B...")

        # 4-bit quantization for RTX 4060 8GB VRAM
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quant,
                device_map="auto",
                dtype=torch.float16,
            )
        )

        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        print("Qwen model loaded successfully!")

    def ask(self, image_path, instruction, max_new_tokens=256):
        # Resolve image path
        path = Path(image_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing science diagram: {path}"
            )

        # FIX: Open image directly instead of using path.as_uri()
        with Image.open(path) as img:
            image = img.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        # Prepare chat template
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process image and text
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Generate response
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Remove input tokens from generated output
        new_tokens = generated[:, inputs.input_ids.shape[1]:]

        response = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()
