import torch
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CKPT = "transformersbook/bert-base-uncased-finetuned-clinc"
STUDENT_CKPT = "distilbert-base-uncased"
TEACHER_CKPT = "transformersbook/bert-base-uncased-finetuned-clinc"

NEW_MODEL_CKPT = "zhan/distillbert-base-uncased-finetuned-clinc"
ONNX_MODEL_PATH = Path("onnx/model.onnx")
