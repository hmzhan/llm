from transformers import pipeline
from src.efficient_llm.constants import MODEL_CKPT


pipe = pipeline("text-classification", model=MODEL_CKPT)
