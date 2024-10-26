import torch

MODEL_CKPT = "transformersbook/bert-base-uncased-finetuned-clinc"
STUDENT_CKPT = "distillbert-base-uncased"
TEACHER_CKPT = "transformersbook/bert-base-uncased-finetuned-clinc"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

