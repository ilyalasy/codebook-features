"""Train script for Codebook models."""

import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field, replace
from typing import Optional

import hydra
import omegaconf
import pandas as pd
import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM

import wandb
from codebook_features import models, run_clm
from codebook_features import trainer as cb_trainer

logger = logging.getLogger(__name__)


@dataclass
class ModelConfigArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""

    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    continue_training: bool = field(default=False)
    model_type: str = field(default="gptneox")
    hidden_size: int = field(default=128)
    intermediate_size: int = field(default=512)
    num_hidden_layers: int = field(default=4)
    num_attention_heads: int = field(default=4)
    rotary_emb_base: int = field(default=10000)
    seq_len: int = field(default=128)
    vocab_size: int = field(default=11)


# shortened arg names to compress wandb titles
shortened_args = {
    "model_name_or_path": "mod",
    "learning_rate": "lr",
    "per_device_train_batch_size": "bs",
    "codebook_type": "cbt",
    "num_codes": "cbs",
    "num_codebooks": "ncb",
    "layers_to_snap": "cb_layers",
    "similarity_metric": "sim",
    "codebook_at": "cb_at",
    "loss": "loss",
    "train_model_params": "train_mod",
    "model_lr_factor": "mod_lrf",
    "k_codebook": "k",
    "dataset_name": "ds",
}


def prepare_logging(cfg):
    """Prepare log config and tags for wandb."""
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    flat_cfg_dict = pd.json_normalize(cfg_dict, sep="@").to_dict(orient="records")[0]
    flat_cfg_dict = {k.split("@")[-1]: v for k, v in flat_cfg_dict.items()}

    # prepare tags and wandb run name from tags
    tags = sorted(cfg.tags)
    for key in sorted(cfg.tag_keys):
        tags.append(f"{shortened_args[key]}: {flat_cfg_dict[key]}")
    if tags:
        cfg_dict["training_args"]["run_name"] = ", ".join(tags)

    return cfg_dict, tags


def get_baseline(training_args, model_args, data_args, model):
    """Get baseline metrics for the original model (no codebooks applied)."""
    baseline_output_dir = training_args.output_dir + "_baseline"
    eval_args = replace(
        training_args,
        output_dir=baseline_output_dir,
    )
    trainer, lm_datasets, _, last_checkpoint = run_clm.get_trainer_and_dataset(
        model_args,
        data_args,
        eval_args,
        model,
    )
    model = torch.compile(model)
    baseline_metrics = run_clm.run_trainer(model_args, data_args, training_args, trainer, lm_datasets, last_checkpoint)
    baseline_metrics = {"baseline/" + k: v for k, v in baseline_metrics.items()}
    with open(baseline_output_dir + "/metrics.json", "w") as f:
        json.dump(baseline_metrics, f)
    return baseline_metrics


def get_optimizer(training_args, model):
    """Get optimizer for codebook based models.

    Returns different optimizers based on whether the model params are being trained or not.
    """
    if training_args.train_model_params:
        params = [
            {
                "params": model.get_codebook_params(),
                "lr": training_args.learning_rate,
                # weight decay for codebook params is used through
                # `codebook_weight_decay` param that is used directly
                # to compute regularized loss.
                "weight_decay": 0.0,
            },
            {
                "params": model.get_model_params(),
                "lr": training_args.model_lr_factor * training_args.learning_rate,
                "weight_decay": training_args.weight_decay,
            },
        ]
    else:
        params = model.get_codebook_params()
    if len(params) > 0:
        optimizer = torch.optim.AdamW(
            params,
            training_args.learning_rate,
        )
    else:
        RuntimeWarning("Codebook not found in model. Training with model params.")
        optimizer = None
    return optimizer


def load_model(config_args: ModelConfigArguments):
    """Load the model based on the config."""
    if config_args.model_path is not None and config_args.continue_training:
        model = transformers.AutoModelForCausalLM.from_pretrained(config_args.model_path)
    elif config_args.model_path is not None:
        config = transformers.AutoConfig.from_pretrained(config_args.model_path)
        model = transformers.AutoModelForCausalLM.from_config(config)
        logger.info("Initialized model from config.")
    elif config_args.model_type == "gptneox":
        config = GPTNeoXConfig(
            vocab_size=config_args.vocab_size,
            hidden_size=config_args.hidden_size,
            num_hidden_layers=config_args.num_hidden_layers,
            num_attention_heads=config_args.num_attention_heads,
            intermediate_size=config_args.intermediate_size,
            rotary_emb_base=config_args.rotary_emb_base,
            bos_token_id=config_args.vocab_size - 1,
            eos_token_id=config_args.vocab_size - 1,
            max_position_embeddings=config_args.seq_len,
        )
        model = GPTNeoXForCausalLM(config=config)
    elif config_args.model_type == "gpt2":
        config = GPT2Config(
            vocab_size=config_args.vocab_size,
            n_embd=config_args.hidden_size,
            n_layer=config_args.num_hidden_layers,
            n_head=config_args.num_attention_heads,
            n_inner=config_args.intermediate_size,
            bos_token_id=config_args.vocab_size - 1,
            eos_token_id=config_args.vocab_size - 1,
            max_position_embeddings=config_args.seq_len,
        )
        model = GPT2LMHeadModel(config=config)
    else:
        raise ValueError(f"Unknown model type {config_args.model_type}")
    return model


@hydra.main(config_path="config", config_name="main_tinystories", version_base=None)
def main(cfg):
    """Train codebook based models parametrized using hydra.

    Args:
        cfg: hydra config.

    Returns: metrics for the trained model.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    cfg_dict, tags = prepare_logging(cfg)
    training_args = run_clm.TrainingArguments(
        **(cfg_dict["training_args"]),
        local_rank=local_rank,
    )

    model_args = run_clm.ModelArguments(**cfg.get("model_args", {}))
    data_args = run_clm.DataTrainingArguments(**cfg.data_args)
    config_args = ModelConfigArguments(**cfg.model_config_args)

    wandb_initilized = False
    if training_args.local_rank <= 0 and "wandb" in training_args.report_to:
        wandb.init(
            project=cfg.project,
            name=training_args.run_name,
            tags=tags,
            settings=wandb.Settings(code_dir="."),
            config=cfg_dict,
            dir=training_args.output_dir,
        )
        wandb_initilized = True
        training_args.output_dir = pathlib.Path(wandb.run.dir).parent / "train_output"
        training_args.output_dir = str(training_args.output_dir)

    model = load_model(config_args)

    if cfg.get_baseline:
        return get_baseline(training_args, model_args, data_args, model)

    if cfg_dict["apply_codebook"]:
        codebook_config = models.CodebookModelConfig(**cfg_dict["codebook_args"])
        model = models.wrap_codebook(
            model_or_path=model,
            config=codebook_config,
            pretrained_path=cfg.pretrained_path,
        )

        if cfg.enable_logging:
            model.enable_logging()

    optimizer = get_optimizer(training_args, model)

    callbacks = [cb_trainer.WandbCallback()] if wandb_initilized else []
    if cfg.k_scheduler_kwargs is not None:
        k_scheduler = cb_trainer.MulticodeKScheduler(k_min=cfg.codebook_args.k_codebook, **cfg.k_scheduler_kwargs)
        callbacks.append(k_scheduler)

    trainer, lm_datasets, _, last_checkpoint = run_clm.get_trainer_and_dataset(
        model_args,
        data_args,
        training_args,
        model,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )

    if codebook_config.kmeans_init and training_args.local_rank <= 0:
        model.init_codebook(trainer.get_train_dataloader())

    model.enable_codebooks()
    # compile doesn't work on Windows or python 3.11+ currently
    if os.name != "nt" and sys.version_info < (3, 11):
        model = torch.compile(model)
    metrics = run_clm.run_trainer(model_args, data_args, training_args, trainer, lm_datasets, last_checkpoint)

    return metrics


if __name__ == "__main__":
    main()
