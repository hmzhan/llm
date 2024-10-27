import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CKPT = "transformersbook/bert-base-uncased-finetuned-clinc"
STUDENT_CKPT = "distilbert-base-uncased"
TEACHER_CKPT = "transformersbook/bert-base-uncased-finetuned-clinc"


