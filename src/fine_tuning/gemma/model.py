import torch
from transformers import GemmaTokenizerFast
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType
)
from src.fine_tuning.gemma.constants import Config

config = Config()

lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    # only target self-attention
    target_modules=["q_proj", "k_proj", "v_proj"],
    layers_to_transform=[i for i in range(42) if i >= config.freeze_layers],
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
    task_type=TaskType.SEQ_CLS,
)

tokenizer = GemmaTokenizerFast.from_pretrained(config.model_ckpt)
tokenizer.add_eos_token = True  # add <eos> at the end
tokenizer.padding_side = "right"


model = Gemma2ForSequenceClassification.from_pretrained(
    config.model_ckpt,
    num_labels=3,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
