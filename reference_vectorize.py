import sys
import os
import json
import time

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM


def use_local_1b_decoder(model_path):
    """Construct the decoder whose weights are already in the public checkpoint."""
    from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM
    from starvector.model.llm.starcoder import StarCoderModel

    def initialize(self, config, **kwargs):
        torch.nn.Module.__init__(self)
        self.init_tokenizer(model_path)
        self.max_length = config.max_length
        decoder_config = GPTBigCodeConfig(
            vocab_size=len(self.tokenizer), n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers, n_head=config.num_attention_heads,
            n_positions=config.max_position_embeddings,
            multi_query=config.multi_query, activation_function="gelu_pytorch_tanh",
            layer_norm_epsilon=1e-5, use_cache=config.use_cache,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        self.transformer = GPTBigCodeForCausalLM(decoder_config)
        self.prompt = "<svg"

    StarCoderModel.__init__ = initialize


def main(image_path, output_path, model_path):
    started = time.monotonic()
    precision = os.environ.get("VECTOR_REFERENCE_PRECISION", "float16")
    if precision not in ("float16", "bfloat16"):
        raise ValueError("Unsupported reference precision")
    dtype = getattr(torch, precision)
    with open(os.path.join(model_path, "config.json")) as source:
        config = json.load(source)
    is_8b = config["starcoder_model_name"] == "bigcode/starcoder2-7b"
    repository = "starvector/starvector-8b-im2svg" if is_8b else "starvector/starvector-1b-im2svg"
    hf_hub_download(repository, "starvector_arch.py", local_dir=model_path)
    if not is_8b:
        use_local_1b_decoder(model_path)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
        output_loading_info=True,
    )
    if not is_8b:
        # The public 1B safetensors index stores wte.weight only. Re-establish
        # the decoder's shared output head after the outer model loads it.
        model.model.svg_transformer.transformer.tie_weights()
    model = model.cuda().eval()
    diagnostic_path = os.environ.get("VECTOR_DEBUG_RAW_OUTPUT")
    diagnostics = {"loading_info": loading_info, "load_seconds": time.monotonic() - started}
    if not is_8b:
        decoder = model.model.svg_transformer.transformer
        diagnostics["output_head_tied"] = (
            decoder.lm_head.weight.data_ptr() == decoder.transformer.wte.weight.data_ptr()
        )
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
    image = image.to(dtype).cuda()
    diagnostics.update(image_shape=list(image.shape), image_dtype=str(image.dtype),
                       image_min=image.min().item(), image_max=image.max().item())
    save_diagnostics()
    generation_started = time.monotonic()
    with torch.inference_mode():
        svg = model.generate_im2svg(
            {"image": image},
            max_length=16000 if is_8b else 7800,
            min_length=10,
            num_beams=1,
            use_nucleus_sampling=True,
            top_p=0.95,
            temperature=0.7 if is_8b else 0.2,
            repetition_penalty=1.0,
            length_penalty=0.5 if is_8b else 1.0,
        )[0]
    diagnostics.update(generation_seconds=time.monotonic() - generation_started,
                       output_characters=len(svg))
    save_diagnostics()
    with open(output_path, "w", encoding="utf-8") as output:
        output.write(svg)


if __name__ == "__main__":
    main(*sys.argv[1:])
