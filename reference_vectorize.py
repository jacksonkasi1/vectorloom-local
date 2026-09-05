import sys

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM


def main(image_path, output_path, model_path):
    hf_hub_download("starvector/starvector-8b-im2svg", "starvector_arch.py", local_dir=model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda().eval()
    # Keep the same preprocessing path and generation controls used by the
    # upstream StarVector 8B quickstart.  This matters for transparent logos
    # and avoids the model immediately closing an otherwise empty canvas.
    image = model.process_images([Image.open(image_path).convert("RGB")])[0]
    image = image.to(torch.float16).cuda()
    with torch.inference_mode():
        svg = model.generate_im2svg(
            {"image": image},
            max_length=4000,
            temperature=1.5,
            length_penalty=-1,
            repetition_penalty=3.1,
        )[0]
    with open(output_path, "w", encoding="utf-8") as output:
        output.write(svg)


if __name__ == "__main__":
    main(*sys.argv[1:])
