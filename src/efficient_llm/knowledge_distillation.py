import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Trainer,
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from datasets import load_metric
import numpy as np
from .src.efficient_llm.constants import STUDENT_CKPT, TEACHER_CKPT, DEVICE
from .src.efficient_llm.data import clinc

accuracy_score = load_metric("accuracy", trust_remote_code=True)


class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_stu = model(**inputs)
        # Extract cross entropy loss and logits from student
        loss_ce = outputs_stu.loss
        logits_stu = outputs_stu.logits
        # Extract logits from teacher
        with torch.no_grad():
            outputs_tea = self.teacher_model(**inputs)
            logits_tea = outputs_tea.logits
        # Soften probabilities and compute distillation loss
        loss_fct = nn.KLDivLoss(reduction='batchmean')
        loss_kd = self.args.temperature ** 2 * loss_fct(
            F.log_softmax(logits_stu / self.args.temperature, dim=-1),
            F.softmax(logits_tea / self.args.temperature, dim=-1)
        )
        # Return weighted student loss
        loss = self.args.alpha * loss_ce + (1. - self.args.alpha) * loss_kd
        return (loss, outputs_stu) if return_outputs else loss


def compute_metrics(pred):
    """
    Compute metrics
    """
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_score.compute(predictions=predictions, references=labels)


def student_init(pipe):
    student_config = AutoConfig.from_pretrained(
        STUDENT_CKPT,
        num_labels=clinc["test"].features["intent"].num_classes,
        id2label=pipe.model.config.id2label,
        label2id=pipe.model.config.label2id
    )
    return AutoModelForSequenceClassification.from_pretrained(STUDENT_CKPT, config=student_config).to(DEVICE)


def tokenize_text(batch):
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_CKPT)
    return student_tokenizer(batch["text"], truncation=True)


def knowledge_distillation():
    # student toknizer
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_CKPT)
    # teacher model
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_CKPT, num_labels=clinc["test"].features["intent"].num_classes).to(DEVICE)
    # dataset
    clinc_enc = clinc.map(tokenize_text, batched=True, remove_columns=["text"])
    clinc_enc = clinc_enc.rename_column("intent", "labels")
    # training args
    training_args = DistillationTrainingArguments(
        output_dir="distillbert-base-uncased-finetuned-clinc",
        evaluation_strategy="epoch",
        num_train_epochs=10,
        temperature=7,
        learning_rate=2e-5,
        per_device_train_batch_size=48,
        per_device_eval_batch_size=48,
        alpha=0.12,
        weight_decay=0.01,
        push_to_hub=True,
        report_to="none"
    )
    # Trainer
    return DistillationTrainer(
        model_init=student_init,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=clinc_enc['train'],
        eval_dataset=clinc_enc['validation'],
        compute_metrics=compute_metrics,
        tokenizer=student_tokenizer
    )
