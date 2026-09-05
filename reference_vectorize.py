import sys
import os
import json
import time

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM


def main(image_path, output_path, model_path):
    started = time.monotonic()
    hf_hub_download("starvector/starvector-8b-im2svg", "starvector_arch.py", local_dir=model_path)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True,
        output_loading_info=True,
    )
    model = model.cuda().eval()
    diagnostic_path = os.environ.get("VECTOR_DEBUG_RAW_OUTPUT")
    diagnostics = {"loading_info": loading_info, "load_seconds": time.monotonic() - started}
    def save_diagnostics():
        if diagnostic_path:
            with open(diagnostic_path + ".json", "w") as output:
                json.dump(diagnostics, output, default=str)
    save_diagnostics()
    # Match the released 8B image-to-SVG evaluation configuration.  The
    # library defaults are intentionally generic (two beams, temperature 1.0,
    # and a 30-token limit), while the published checkpoint is evaluated with
    # nucleus sampling and the controls below.
    image = model.process_images([Image.open(image_path).convert("RGB")])[0]
    image = image.to(torch.float16).cuda()
    diagnostics.update(image_shape=list(image.shape), image_dtype=str(image.dtype),
                       image_min=image.min().item(), image_max=image.max().item())
    save_diagnostics()
    generation_started = time.monotonic()
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
    diagnostics.update(generation_seconds=time.monotonic() - generation_started,
                       output_characters=len(svg))
    save_diagnostics()
    with open(output_path, "w", encoding="utf-8") as output:
        output.write(svg)


if __name__ == "__main__":
    main(*sys.argv[1:])
