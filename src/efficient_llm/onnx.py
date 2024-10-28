import os
import numpy as np
from pathlib import Path
from psutil import cpu_count
from transformers.convert_graph_to_onnx import convert
from transformers import AutoTokenizer
from scipy.special import softmax
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions
)
from src.efficient_llm.model_performance import PerformanceBenchmark
from src.efficient_llm.data import clinc


def convert_model_onnx(model_ckpt, onnx_model_path):
    os.environ['OMP_NUM_THREADS'] = f"{cpu_count()}"
    os.environ['OMP_WAIT_POLICY'] = "ACTIVE"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    convert(
        framework="pt",
        model=model_ckpt,
        tokenizer=tokenizer,
        output=onnx_model_path,
        opset=12,
        pipeline_name="text-classification"
    )


def create_model_for_provider(model_path, provider="CPUExecutionProvider"):
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(str(model_path), options, providers=[provider])
    session.disable_fallback()
    return session


class OnnxPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, query):
        model_inputs = self.tokenizer(query, return_tensors="pt")
        inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
        logits = self.model.run(None, inputs_onnx)[0][0, :]
        probs = softmax(logits)
        pred_idx = np.argmax(probs).item()
        return [{
            "label": clinc["test"].features["intent"].int2str(pred_idx),
            "score": probs[pred_idx]

        }]


class OnnxPerformanceBenchmark(PerformanceBenchmark):
    def __init__(self, *args, model_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path

    def compute_size(self):
        size_mb = Path(self.model_path).stat().st_size / (1024 * 1024)
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": size_mb}
