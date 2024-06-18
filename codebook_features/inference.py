from pathlib import Path


import hydra
import torch
import transformers

from codebook_features import models
from codebook_features.train_codebook import ModelConfigArguments, load_model, prepare_logging

PRETRAINED_PATH = "output_tiny/wandb/latest-run/train_output"
DEVICE = "mps"


@hydra.main(config_path="config", config_name="main_tinystories", version_base=None)
def main(cfg):
    """Train codebook based models parametrized using hydra.

    Args:
        cfg: hydra config.

    Returns: metrics for the trained model.
    """
    config_args = ModelConfigArguments(**cfg.model_config_args)
    cfg_dict, _ = prepare_logging(cfg)
    model = load_model(config_args)

    codebook_config = models.CodebookModelConfig(**cfg_dict["codebook_args"])

    model = models.wrap_codebook(model_or_path=model, config=codebook_config)
    with open(Path(PRETRAINED_PATH) / "pytorch_model.bin", "rb") as f:
        state = torch.load(f, map_location=DEVICE)
    model.load_state_dict(state)

    model = model.to(DEVICE).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(PRETRAINED_PATH)

    prompt = "Once upon a time there was"

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(DEVICE)
    streamer = transformers.TextStreamer(
        tokenizer,
    )
    output = model.generate(**inputs, streamer=streamer, max_new_tokens=64)
    # output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # print(output_text)


if __name__ == "__main__":
    main()
