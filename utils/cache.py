import os
import shutil
import hashlib
import json
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download
from rich.console import Console


console = Console()


def save_top_documents_texts(results: dict, documents: list, dataset_name: str, top_n: int = 100):
    """Cache the text of top N documents per classifier.
    
    This saves document texts for the highest-scoring documents to avoid
    needing to stream from datasets later during visualization.
    Merges with existing cache to preserve texts from previous runs.
    
    Args:
        results: Dictionary mapping classifier names to list of score dictionaries
        documents: List of document dictionaries (with 'id' and 'text' fields)
        dataset_name: Name of the dataset (e.g., 'fineweb', 'fineweb-edu')
        top_n: Number of top documents to cache per classifier (default: 100)
    """
    console.log(f"[bold cyan]Caching top {top_n} document texts per classifier...[/bold cyan]")
    
    cache_dir = Path("cache") / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "top_documents_texts.json"
    
    existing_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                existing_cache = json.load(f)
            console.log(f"[green]Loaded {len(existing_cache)} existing cached texts[/green]")
        except Exception as e:
            console.log(f"[yellow]Could not load existing cache: {e}[/yellow]")
    
    doc_id_to_text = {doc['id']: doc['text'] for doc in documents}
    top_docs_cache = existing_cache.copy()
    new_texts_added = 0
    
    for clf_name, scores in results.items():
        sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)[:top_n]
        console.log(f"[yellow]Processing top {top_n} documents for {clf_name}...[/yellow]")
        
        for score_data in sorted_scores:
            doc_id = score_data['id']
            if doc_id not in top_docs_cache and doc_id in doc_id_to_text:
                top_docs_cache[doc_id] = doc_id_to_text[doc_id]
                new_texts_added += 1
    
    with open(cache_file, 'w') as f:
        json.dump(top_docs_cache, f, indent=2)
    
    console.log(f"[bold green]Cached {len(top_docs_cache)} total document texts ({new_texts_added} new) to {cache_file}[/bold green]")


def download_fasttext_model(hub_repo, hub_filename, local_filename, models_dir="models"):
    """
    Generic utility to download a FastText model from HuggingFace Hub.
    
    Args:
        hub_repo: HuggingFace Hub repository name
        hub_filename: Filename in the Hub repository
        local_filename: Local filename to save as
        models_dir: Directory to save models to
    """
    model_path = os.path.join(models_dir, local_filename)
    if os.path.exists(model_path):
        console.log(f"[green]Model already exists at {model_path}[/green]")
        return model_path
    
    console.log(f"[yellow]Downloading FastText model to {model_path}...[/yellow]")
    os.makedirs(models_dir, exist_ok=True)
    downloaded_path = hf_hub_download(hub_repo, hub_filename)
    shutil.copy(downloaded_path, model_path)
    console.log(f"[green]Model downloaded to {model_path}.[/green]")
    return model_path


def download_transformer_model(hub_name, local_dirname, models_dir="models", trust_remote_code=False, torch_dtype=None):
    """
    Generic utility to download a Transformer model from HuggingFace Hub.
    
    Args:
        hub_name: HuggingFace Hub model name
        local_dirname: Local directory name to save as
        models_dir: Base directory to save models to
        trust_remote_code: Whether to trust remote code
        torch_dtype: Optional torch dtype for the model
    
    Returns:
        Path to the downloaded model directory
    """
    model_dir = os.path.join(models_dir, local_dirname)
    
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        console.log(f"[green]Model already exists at {model_dir}[/green]")
        return model_dir
    
    console.log(f"[yellow]Downloading transformer model to {model_dir}...[/yellow]")
    os.makedirs(models_dir, exist_ok=True)
    
    model_kwargs = {}
    if trust_remote_code:
        model_kwargs['trust_remote_code'] = True
    if torch_dtype:
        model_kwargs['torch_dtype'] = torch_dtype
    
    tokenizer = AutoTokenizer.from_pretrained(hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hub_name, **model_kwargs)
    
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    console.log(f"[green]Model downloaded to {model_dir}.[/green]")
    return model_dir


class DocumentClassifier(ABC):
    
    def __init__(self, config=None):
        dataset_name = "fineweb"
        if config and "dataset_name" in config:
            dataset_name = config["dataset_name"]
        
        cache_dir = Path("cache") / dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / f"{self.__class__.__name__}.json"
        self._cache = self._load_cache()
    
    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self._cache, f)
    
    @abstractmethod
    def _score_documents(self, documents):
        pass
    
    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            console.log("[green]Using CUDA for inference.[/green]")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            console.log("[green]Using MPS for inference.[/green]")
        else:
            device = torch.device("cpu")
            console.log("[yellow]Using CPU for inference.[/yellow]")
        return device
    
    def _load_transformer_model(self, model_dir, hub_name, trust_remote_code=False, torch_dtype=None):
        model_kwargs = {}
        if trust_remote_code:
            model_kwargs['trust_remote_code'] = True
        if torch_dtype:
            model_kwargs['torch_dtype'] = torch_dtype
        
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            console.log(f"[yellow]Loading model and tokenizer from local {model_dir}...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, **model_kwargs)
        else:
            console.log(f"[yellow]Loading model and tokenizer from HuggingFace Hub ({hub_name})...[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained(hub_name)
            model = AutoModelForSequenceClassification.from_pretrained(hub_name, **model_kwargs)
        
        device = self._get_device()
        model = model.to(device)
        return tokenizer, model, device
    
    def _get_document_hash(self, document):
        content = f"{document['id']}:{document['text']}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def score_documents(self, documents):
        from tqdm import tqdm
        classifier_name = self.__class__.__name__
        console.log(f"[bold cyan]Scoring documents with {classifier_name} (with caching)...[/bold cyan]")
        
        results, docs_to_score = [], []
        cache_hits = cache_misses = 0
        
        for doc in documents:
            doc_hash = self._get_document_hash(doc)
            if doc_hash in self._cache:
                results.append(self._cache[doc_hash])
                cache_hits += 1
            else:
                docs_to_score.append(doc)
                cache_misses += 1
        
        console.log(f"[green]Cache hits: {cache_hits}, Cache misses: {cache_misses}[/green]")
        
        if docs_to_score:
            new_results = self._score_documents(docs_to_score)
            for doc, result in zip(docs_to_score, new_results):
                doc_hash = self._get_document_hash(doc)
                self._cache[doc_hash] = result
                results.append(result)
            self._save_cache()
        
        doc_id_to_idx = {doc['id']: idx for idx, doc in enumerate(documents)}
        results.sort(key=lambda r: doc_id_to_idx[r['id']])
        return results

