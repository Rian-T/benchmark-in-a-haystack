from abc import ABC, abstractmethod
import random
from datasets import load_dataset

class Benchmark(ABC):
    @abstractmethod
    def load_samples(self, count=5, subjects=None):
        pass

    @abstractmethod
    def format_sample(self, sample, subject=None):
        pass

class MMLUBenchmark(Benchmark):
    dataset = "cais/mmlu"
    split = "test"
    format_template = "Subject: {subject}\nQuestion: {question}\n{choices}\nAnswer: {answer}"

    def load_samples(self, count=5, subjects=None):
        samples = []
        if not subjects:
            raise ValueError("MMLU requires subjects")
        for subject in subjects:
            dataset = load_dataset(self.dataset, subject, split=self.split)
            samples.append({
                "subject": subject,
                "data": dataset[0],
                "benchmark_type": "mmlu"
            })
        return samples

    def format_sample(self, sample, subject=None):
        data = sample["data"]
        question = data["question"]
        answer = chr(65 + data["answer"])
        choices = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(data["choices"])])
        subject = subject or sample.get("subject")
        return self.format_template.format(subject=subject, question=question, choices=choices, answer=answer)

class GSM8KBenchmark(Benchmark):
    dataset = "openai/gsm8k"
    name = "main"
    split = "test"
    format_template = "Math Problem: {question}\n\nSolution: {answer}"

    def load_samples(self, count=5, subjects=None):
        dataset = load_dataset(self.dataset, name=self.name, split=self.split)
        indices = random.sample(range(len(dataset)), count)
        return [{"data": dataset[i], "benchmark_type": "gsm8k"} for i in indices]

    def format_sample(self, sample, subject=None):
        data = sample["data"]
        return self.format_template.format(question=data["question"], answer=data["answer"])

class GPQABenchmark(Benchmark):
    dataset = "hendrydong/gpqa_diamond"
    split = "test"
    format_template = "Problem:\n{problem}\n\nSolution:\n{solution}"

    def load_samples(self, count=5, subjects=None):
        dataset = load_dataset(self.dataset, split=self.split)
        indices = random.sample(range(len(dataset)), count)
        return [{"data": dataset[i], "benchmark_type": "gpqa"} for i in indices]

    def format_sample(self, sample, subject=None):
        data = sample["data"]
        return self.format_template.format(problem=data["problem"], solution=data["solution"])

# Registry for easy extensibility
BENCHMARKS = {
    "mmlu": MMLUBenchmark(),
    "gsm8k": GSM8KBenchmark(),
    "gpqa": GPQABenchmark(),
}
