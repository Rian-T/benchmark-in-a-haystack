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
            for idx in range(count):
                samples.append({
                    "subject": subject,
                    "data": dataset[idx],
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

class ARCChallengeBenchmark(Benchmark):
    dataset = "allenai/ai2_arc"
    config = "ARC-Challenge"
    split = "test"
    format_template = "Question: {question}\n{choices}\nAnswer: {answer}"

    def load_samples(self, count=5, subjects=None):
        dataset = load_dataset(self.dataset, self.config, split=self.split)
        indices = random.sample(range(len(dataset)), min(count, len(dataset)))
        return [{"data": dataset[i], "benchmark_type": "arc_challenge"} for i in indices]

    def format_sample(self, sample, subject=None):
        data = sample["data"]
        choices = "\n".join([f"{label}. {text}" for label, text in zip(data['choices']['label'], data['choices']['text'])])
        return self.format_template.format(question=data["question"], choices=choices, answer=data["answerKey"])

class ARCEasyBenchmark(Benchmark):
    dataset = "allenai/ai2_arc"
    config = "ARC-Easy"
    split = "test"
    format_template = "Question: {question}\n{choices}\nAnswer: {answer}"

    def load_samples(self, count=5, subjects=None):
        dataset = load_dataset(self.dataset, self.config, split=self.split)
        indices = random.sample(range(len(dataset)), min(count, len(dataset)))
        return [{"data": dataset[i], "benchmark_type": "arc_easy"} for i in indices]

    def format_sample(self, sample, subject=None):
        data = sample["data"]
        choices = "\n".join([f"{label}. {text}" for label, text in zip(data['choices']['label'], data['choices']['text'])])
        return self.format_template.format(question=data["question"], choices=choices, answer=data["answerKey"])

class HellaSwagBenchmark(Benchmark):
    dataset = "Rowan/hellaswag"
    split = "validation"
    format_template = "Context: {context}\n\nChoose the most plausible continuation:\n{endings}\nAnswer: {answer}"

    def load_samples(self, count=5, subjects=None):
        dataset = load_dataset(self.dataset, split=self.split)
        indices = random.sample(range(len(dataset)), min(count, len(dataset)))
        return [{"data": dataset[i], "benchmark_type": "hellaswag"} for i in indices]

    def format_sample(self, sample, subject=None):
        data = sample["data"]
        endings = "\n".join([f"{chr(65+i)}. {ending}" for i, ending in enumerate(data['endings'])])
        answer = chr(65 + int(data['label']))
        return self.format_template.format(context=data["ctx"], endings=endings, answer=answer)

class PIQABenchmark(Benchmark):
    dataset = "gimmaru/piqa"
    split = "validation"
    format_template = "Goal: {goal}\n\nWhich solution is better?\nA. {sol1}\nB. {sol2}\nAnswer: {answer}"

    def load_samples(self, count=5, subjects=None):
        dataset = load_dataset(self.dataset, split=self.split)
        indices = random.sample(range(len(dataset)), min(count, len(dataset)))
        return [{"data": dataset[i], "benchmark_type": "piqa"} for i in indices]

    def format_sample(self, sample, subject=None):
        data = sample["data"]
        answer = chr(65 + data['label'])
        return self.format_template.format(goal=data["goal"], sol1=data["sol1"], sol2=data["sol2"], answer=answer)

class TruthfulQABenchmark(Benchmark):
    dataset = "truthful_qa"
    config = "generation"
    split = "validation"
    format_template = "Question: {question}\n\nBest Answer: {best_answer}\n\nCorrect Answers:\n{correct_answers}"

    def load_samples(self, count=5, subjects=None):
        dataset = load_dataset(self.dataset, self.config, split=self.split)
        indices = random.sample(range(len(dataset)), min(count, len(dataset)))
        return [{"data": dataset[i], "benchmark_type": "truthfulqa"} for i in indices]

    def format_sample(self, sample, subject=None):
        data = sample["data"]
        correct_answers = "\n".join([f"- {ans}" for ans in data['correct_answers']])
        return self.format_template.format(
            question=data["question"], 
            best_answer=data["best_answer"],
            correct_answers=correct_answers
        )

# Registry for easy extensibility
BENCHMARKS = {
    "mmlu": MMLUBenchmark(),
    "gsm8k": GSM8KBenchmark(),
    "gpqa": GPQABenchmark(),
    "arc_challenge": ARCChallengeBenchmark(),
    "arc_easy": ARCEasyBenchmark(),
    "hellaswag": HellaSwagBenchmark(),
    "piqa": PIQABenchmark(),
    "truthfulqa": TruthfulQABenchmark(),
}
