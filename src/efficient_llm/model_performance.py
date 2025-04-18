import torch
import numpy as np
from pathlib import Path
from time import perf_counter
from datasets import load_metric


accuracy_score = load_metric("accuracy", trust_remote_code=True)


class PerformanceBenchmark:
    def __init__(self, pipeline, dataset, optim_type="BERT baseline"):
        self.pipeline = pipeline
        self.dataset = dataset
        self.optim_type = optim_type

    def compute_accuracy(self):
        """
        Compute model prediction accuracy
        """
        preds, labels = [], []
        intents = self.dataset.features["intent"]
        for example in self.dataset:
            pred = self.pipeline(example["text"])[0]["label"]
            label = example["intent"]
            preds.append(intents.str2int(pred))
            labels.append(label)
        accuracy = accuracy_score.compute(predictions=preds, references=labels)
        print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
        accuracy['accuracy'] = round(accuracy['accuracy'], 3)
        return accuracy

    def compute_size(self):
        """
        Compute model size
        """
        state_dict = self.pipeline.model.state_dict()  # all model info including bias, and weights for each layer
        tmp_path = Path("model.pt")
        torch.save(state_dict, tmp_path)
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)  # model size in mb
        tmp_path.unlink()
        print(f"Model size (MB) - {size_mb:.2f}")
        return {"size_mb": round(size_mb, 3)}

    def time_pipeline(self, query="What is the pin number for my account?"):
        """
        Estimate average time the model takes to generate outputs.
        """
        latencies = []
        for _ in range(10):
            _ = self.pipeline(query)
        # Timed run
        for _ in range(100):
            start_time = perf_counter()
            _ = self.pipeline(query)
            latency = perf_counter() - start_time
            latencies.append(latency)
        # Compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies)
        print(f"Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f}")
        return {"time_avg_ms": round(time_avg_ms, 3), "time_std_ms": round(time_std_ms, 3)}

    def run_benchmark(self):
        """
        Run code to get information about model size, accuracy, and latency
        """
        metrics = dict()
        metrics[self.optim_type] = self.compute_size()
        metrics[self.optim_type].update(self.time_pipeline())
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics
