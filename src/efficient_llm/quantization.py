import torch
from torch import nn
from torch.quantization import quantize_dynamic
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


def quantization_model(model_ckpt):
    """
    Dynamic quantization of a LLM
    """
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt).to("cpu")
    model_quantized = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    pipe = pipeline("text-classification", model=model_quantized, tokenizer=tokenizer)
    return pipe
