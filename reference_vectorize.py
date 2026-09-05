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
    processor = model.model.processor
    # SigLIP expects a three-channel image.  Flatten transparent uploads onto
    # RGB before preprocessing so PNG logos with an alpha channel work too.
    image = processor(Image.open(image_path).convert("RGB"), return_tensors="pt")["pixel_values"].cuda()
    if image.shape[0] == 1:
        image = image.squeeze(0)
    with torch.inference_mode():
        svg = model.generate_im2svg({"image": image}, max_length=4000)[0]
    with open(output_path, "w", encoding="utf-8") as output:
        output.write(svg)


if __name__ == "__main__":
    main(*sys.argv[1:])
