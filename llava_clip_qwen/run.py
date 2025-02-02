import copy
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    DataCollatorForSeq2Seq,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
)

from llava_clip_qwen.custom_trainer import WebTrainer
from llava_clip_qwen.build_data import LlavaDataset, TrainLLavaModelCollator
from llava_clip_qwen.util import print_trainable_parameters

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="build_llava/qwen")
    train_type: Optional[str] = field(
        default="none",
        metadata={
            "help": """
            1. use_lora:使用lora训练,
            2. none:全量参数训练;
            3. freeze_vision:只冻结vision_tower进行训练
            """
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


def load_model_processor(modelargs: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(modelargs.model_name_or_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,)
    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)

    if modelargs.train_type == "use_lora":
        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model

        LORA_R = 32
        # LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )
        model = get_peft_model(model, config)
        # model.print_trainable_parameters()

    elif modelargs.train_type == "none":
        logging.warning("使用全量参数进行训练")

        pass
    elif modelargs.train_type == "freeze_vision":
        logging.warning("冻结clip.vision_tower,对qwen进行训练")

        for param in model.vision_tower.parameters():
            param.requires_grad = False
    print_trainable_parameters(model)

    return model, processor


def load_dataset_collator(processor, data_args: DataArguments):

    llava_dataset = LlavaDataset(data_args.data_path  )
    data_collator = TrainLLavaModelCollator(processor, -100)

    return llava_dataset, data_collator


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
