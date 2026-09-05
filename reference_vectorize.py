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
    # Match the released 8B image-to-SVG evaluation configuration.  The
    # library defaults are intentionally generic (two beams, temperature 1.0,
    # and a 30-token limit), while the published checkpoint is evaluated with
    # nucleus sampling and the controls below.
    image = model.process_images([Image.open(image_path).convert("RGB")])[0]
    image = image.to(torch.float16).cuda()
    with torch.inference_mode():
        svg = model.generate_im2svg(
            {"image": image},
            max_length=16000,
            min_length=10,
            num_beams=1,
            use_nucleus_sampling=True,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.0,
            length_penalty=0.5,
        )[0]
    with open(output_path, "w", encoding="utf-8") as output:
        output.write(svg)


if __name__ == "__main__":
    main(*sys.argv[1:])
