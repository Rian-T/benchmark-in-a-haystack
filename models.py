import os
import requests
from abc import ABC, abstractmethod
from utils import score_documents
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from tqdm import tqdm


console = Console()

class DocumentClassifier(ABC):
    @abstractmethod
    def score_documents(self, documents):
        pass


class TransformersClassifier(DocumentClassifier, ABC):
    def __init__(
        self,
        *,
        local_model_dir=None,
        hf_model_id=None,
        batch_size=100,
        trust_remote_code=False,
        tokenizer_load_kwargs=None,
        model_load_kwargs=None,
        tokenizer_call_kwargs=None,
        padding="longest",
        truncation=True,
        max_length=None,
    ):
        console.log(f"[bold cyan]Initializing {self.__class__.__name__}...[/bold cyan]")
        self.local_model_dir = local_model_dir
        self.hf_model_id = hf_model_id
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.tokenizer_load_kwargs = tokenizer_load_kwargs or {}
        self.model_load_kwargs = model_load_kwargs or {}
        self.tokenizer_call_kwargs = tokenizer_call_kwargs or {}
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.tokenizer, self.model, self.device = self._load_model()

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        model_dir = self.local_model_dir
        tokenizer_kwargs = dict(self.tokenizer_load_kwargs)
        model_kwargs = dict(self.model_load_kwargs)
        if "trust_remote_code" not in tokenizer_kwargs:
            tokenizer_kwargs["trust_remote_code"] = self.trust_remote_code
        if "trust_remote_code" not in model_kwargs:
            model_kwargs["trust_remote_code"] = self.trust_remote_code

        if model_dir and os.path.exists(model_dir) and os.path.isdir(model_dir):
            console.log(f"[yellow]Loading {self.__class__.__name__} model and tokenizer from local {model_dir}...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained(model_dir, **tokenizer_kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, **model_kwargs)
        elif self.hf_model_id:
            console.log(f"[yellow]Loading {self.__class__.__name__} model and tokenizer from HuggingFace Hub...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id, **tokenizer_kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(self.hf_model_id, **model_kwargs)
        else:
            raise ValueError("Either local_model_dir or hf_model_id must be provided for TransformersClassifier.")

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
        model.eval()
        return tokenizer, model, device

    def _prepare_inputs(self, text_batch):
        tokenizer_kwargs = dict(self.tokenizer_call_kwargs)
        tokenizer_kwargs.setdefault("return_tensors", "pt")
        tokenizer_kwargs.setdefault("padding", self.padding)
        if self.max_length is not None:
            tokenizer_kwargs.setdefault("max_length", self.max_length)
            tokenizer_kwargs.setdefault("truncation", True)
        elif self.truncation is not None:
            tokenizer_kwargs.setdefault("truncation", self.truncation)

        inputs = self.tokenizer(text_batch, **tokenizer_kwargs)
        return inputs.to(self.device)

    def _doc_metadata(self, doc):
        return {
            "id": doc["id"],
            "source": doc["source"],
            "contains_benchmark": doc["contains_benchmark"],
            "benchmark_type": doc["benchmark_type"],
            "benchmark_index": doc.get("benchmark_index", None),
        }

    @abstractmethod
    def _postprocess(self, doc_batch, outputs, inputs):
        """Transform raw model outputs into result dictionaries."""

    def score_documents(self, documents):
        import torch

        console.log(f"[bold cyan]Scoring documents with {self.__class__.__name__}...[/bold cyan]")
        results = []
        for idx_batch in tqdm(range(0, len(documents), self.batch_size)):
            batch_end = min(idx_batch + self.batch_size, len(documents))
            doc_batch = documents[idx_batch:batch_end]
            if not doc_batch:
                continue
            text_batch = [doc["text"] for doc in doc_batch]
            inputs = self._prepare_inputs(text_batch)
            with torch.no_grad():
                outputs = self.model(**inputs)
            results.extend(self._postprocess(doc_batch, outputs, inputs))
        return results

class DCLMClassifier(DocumentClassifier):
    def __init__(self):
        console.log("[bold cyan]Initializing DCLMClassifier...[/bold cyan]")
        self.model = self._load_model()

    @staticmethod
    def _load_model():
        model_path = "models/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
        if not os.path.exists(model_path):
            console.log(f"[yellow]Model not found at {model_path}. Downloading...[/yellow]")
            os.makedirs("models", exist_ok=True)
            url = "https://huggingface.co/mlfoundations/fasttext-oh-eli5/raw/main/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("[green]Downloading FastText model...", total=None)
                response = requests.get(url, stream=True)
                with open(model_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                progress.update(task, completed=1)
            console.log(f"[green]Model downloaded to {model_path}.[/green]")
        from utils import load_fasttext_model
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

class FinewebEduClassifier(TransformersClassifier):
    def __init__(self):
        super().__init__(
            local_model_dir="models/fineweb-edu-classifier",
            hf_model_id="HuggingFaceTB/fineweb-edu-classifier",
            batch_size=100,
        )

    def _postprocess(self, doc_batch, outputs, inputs):
        scores = outputs.logits.detach().cpu().view(-1).tolist()
        results = []
        for doc, score in zip(doc_batch, scores):
            bounded_score = float(score)
            int_score = int(round(max(0, min(bounded_score, 5))))
            result = self._doc_metadata(doc)
            result.update({"score": bounded_score, "int_score": int_score})
            results.append(result)
        return results


class GaperonClassifier(TransformersClassifier):
    def __init__(self):
        super().__init__(
            local_model_dir="models/gaperon-quality-cls",
            hf_model_id="almanach/gaperon-quality-cls",
            batch_size=100,
            trust_remote_code=True,
            max_length=512,
        )

    def _prepare_inputs(self, text_batch):
        inputs = super()._prepare_inputs(text_batch)
        return {k: v[:, :512] for k, v in inputs.items()}

    def _postprocess(self, doc_batch, outputs, inputs):
        import torch.nn.functional as F

        logits_list = outputs.logits_list[0]
        results = []
        for doc, logits in zip(doc_batch, logits_list):
            probs = F.softmax(logits.squeeze(0).detach().cpu().float(), dim=-1).numpy()
            score = float(probs[0] + 0.5 * probs[2])
            int_score = int(round(max(0, min(1 + 2 * score, 3))))
            result = self._doc_metadata(doc)
            result.update({"score": score, "int_score": int_score})
            results.append(result)
        return results


class FinePDFsEduClassifier(TransformersClassifier):
    def __init__(self):
        super().__init__(
            local_model_dir="models/finepdfs-edu-classifier",
            hf_model_id="HuggingFaceFW/finepdfs_edu_classifier_v2_eng_Latn",
            batch_size=16,
            max_length=512,
        )

    def _postprocess(self, doc_batch, outputs, inputs):
        scores = outputs.logits.detach().cpu().view(-1).tolist()
        results = []
        for doc, score in zip(doc_batch, scores):
            bounded_score = float(score)
            int_score = int(round(max(0, min(bounded_score, 5))))
            result = self._doc_metadata(doc)
            result.update({"score": bounded_score, "int_score": int_score})
            results.append(result)
        return results


class FinePDFsDCLMClassifier(TransformersClassifier):
    def __init__(self):
        super().__init__(
            local_model_dir="models/finepdfs-dclm-classifier",
            hf_model_id="HuggingFaceFW/finepdfs_dclm_classifier_eng_Latn",
            batch_size=16,
        )

    def _postprocess(self, doc_batch, outputs, inputs):
        scores = outputs.logits.detach().cpu().view(-1).tolist()
        results = []
        for doc, score in zip(doc_batch, scores):
            bounded_score = float(score)
            int_score = int(round(max(0, min(bounded_score, 5))))
            result = self._doc_metadata(doc)
            result.update({"score": bounded_score, "int_score": int_score})
            results.append(result)
        return results