import os
import requests
from abc import ABC, abstractmethod
from utils import score_documents

# Add rich logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from tqdm import tqdm


console = Console()

class DocumentClassifier(ABC):
    @abstractmethod
    def score_documents(self, documents):
        pass

class DCLMClassifier(DocumentClassifier):
    def __init__(self):
        console.log("[bold cyan]Initializing DCLMClassifier...[/bold cyan]")
        self.model = self._load_model()

    @staticmethod
    def _load_model():
        import shutil
        from huggingface_hub import hf_hub_download
        
        model_path = "models/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
        if not os.path.exists(model_path):
            console.log(f"[yellow]Model not found at {model_path}. Downloading...[/yellow]")
            os.makedirs("models", exist_ok=True)
            # Download using hf_hub_download (proper way to download from HuggingFace)
            downloaded_path = hf_hub_download(
                "mlfoundations/fasttext-oh-eli5",
                "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
            )
            shutil.copy(downloaded_path, model_path)
            console.log(f"[green]Model downloaded to {model_path}.[/green]")
        from utils import load_fasttext_model  # import here to avoid circular import
        return load_fasttext_model(model_path)

    def score_documents(self, documents):
        console.log("[bold cyan]Scoring documents with DCLMClassifier...[/bold cyan]")
        return score_documents(documents, self.model)

class TextbookFastTextClassifier(DocumentClassifier):
    def __init__(self):
        console.log("[bold cyan]Initializing TextbookFastTextClassifier...[/bold cyan]")
        self.model = self._load_model()

    @staticmethod
    def _load_model():
        import fasttext
        import os
        model_path = "models/textbook_model.bin"
        if os.path.exists(model_path):
            console.log(f"[yellow]Loading Textbook FastText model from local {model_path}...[/yellow]")
            return fasttext.load_model(model_path)
        else:
            from huggingface_hub import hf_hub_download
            console.log("[yellow]Loading Textbook FastText model from HuggingFace Hub...[/yellow]")
            return fasttext.load_model(
                hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifer-v1", "model.bin")
            )

    def score_documents(self, documents):
        import re
        from typing import List

        def replace_newlines(text: str) -> str:
            return re.sub("\n+", " ", text)

        console.log("[bold cyan]Scoring documents with TextbookFastTextClassifier...[/bold cyan]")
        texts = [replace_newlines(doc["text"]) for doc in documents]
        preds = self.model.predict(texts)
        # preds: tuple (labels, scores), each is a list of lists
        results = []
        for doc, labels, scores in tqdm(zip(documents, preds[0], preds[1])):
            label = labels[0].lstrip("__label__")
            score = scores[0]
            results.append({
                "id": doc["id"],
                "source": doc["source"],
                "contains_benchmark": doc["contains_benchmark"],
                "benchmark_type": doc["benchmark_type"],
                "benchmark_index": doc.get("benchmark_index", None),
                "score": float(score),
                "label": label
            })
        return results

class FinewebEduClassifier(DocumentClassifier):
    
    def __init__(self):
        console.log("[bold cyan]Initializing FinewebEduClassifier...[/bold cyan]")
        self.tokenizer, self.model, self.device = self._load_model()
        self.batch_size = 100

    @staticmethod
    def _load_model():
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        import os

        model_dir = "models/fineweb-edu-classifier"
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            console.log(f"[yellow]Loading FinewebEduClassifier model and tokenizer from local {model_dir}...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        else:
            console.log("[yellow]Loading FinewebEduClassifier model and tokenizer from HuggingFace Hub...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
            model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
        # Try CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            console.log("[green]Using CUDA for inference.[/green]")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            console.log("[green]Using MPS for inference.[/green]")
        else:
            device = torch.device("cpu")
            console.log("[yellow]Using CPU for inference.[/yellow]")
        model = model.to(device)
        return tokenizer, model, device

    def score_documents(self, documents):
        import torch
        console.log("[bold cyan]Scoring documents with FinewebEduClassifier...[/bold cyan]")
        results = []
        for idx_batch in tqdm(range(0, len(documents), self.batch_size)):
            doc_batch = documents[idx_batch:idx_batch + self.batch_size]
            text_batch = [doc["text"] for doc in doc_batch]
            inputs = self.tokenizer(text_batch, return_tensors="pt", padding="longest", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            for i_doc, doc in enumerate(doc_batch):
                logits = outputs.logits[i_doc].float().detach().cpu().numpy()
                score = logits.item()
                int_score = int(round(max(0, min(score, 5))))
                results.append({
                    "id": doc["id"],
                    "source": doc["source"],
                    "contains_benchmark": doc["contains_benchmark"],
                    "benchmark_type": doc["benchmark_type"],
                    "benchmark_index": doc.get("benchmark_index", None),
                    "score": float(score),
                    "int_score": int_score
                })
        return results


class GaperonClassifier(DocumentClassifier):

    def __init__(self):
        console.log("[bold cyan]Initializing GaperonClassifier...[/bold cyan]")
        self.tokenizer, self.model, self.device = self._load_model()
        self.batch_size = 100

    @staticmethod
    def _load_model():
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        import os

        model_dir = "models/gaperon-quality-cls"
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            console.log(f"[yellow]Loading GaperonClassifier model and tokenizer from local {model_dir}...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
        else:
            console.log("[yellow]Loading GaperonClassifier model and tokenizer from HuggingFace Hub...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained("almanach/gaperon-quality-cls")
            model = AutoModelForSequenceClassification.from_pretrained(
                "almanach/gaperon-quality-cls", trust_remote_code=True
            )
        # Try CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            console.log("[green]Using CUDA for inference.[/green]")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            console.log("[green]Using MPS for inference.[/green]")
        else:
            device = torch.device("cpu")
            console.log("[yellow]Using CPU for inference.[/yellow]")
        model = model.to(device)
        return tokenizer, model, device

    def score_documents(self, documents):
        import torch
        console.log("[bold cyan]Scoring documents with GaperonClassifier...[/bold cyan]")
        results = []
        for idx_batch in tqdm(range(0, len(documents), self.batch_size)):
            doc_batch = documents[idx_batch:idx_batch + self.batch_size]
            text_batch = [doc["text"] for doc in doc_batch]
            inputs = self.tokenizer(text_batch, return_tensors="pt", padding="longest", truncation=True, max_length=512).to(self.device)
            inputs = {k: v[:, :512] for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

            for i_doc, doc in enumerate(doc_batch):
                logits = outputs.logits_list[0][i_doc].squeeze(0).float().softmax(-1).detach().cpu().numpy()
                score = (logits[0] + 0.5 * logits[2]).item()
            # print(score)
                int_score = int(round(max(0, min(1+2*score, 3))))
                results.append({
                    "id": doc["id"],
                    "source": doc["source"],
                    "contains_benchmark": doc["contains_benchmark"],
                    "benchmark_type": doc["benchmark_type"],
                    "benchmark_index": doc.get("benchmark_index", None),
                    "score": float(score),
                    "int_score": int_score
                })
        return results