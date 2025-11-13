import os
import hashlib
import json
import sqlite3
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rich.console import Console


console = Console()


class DocumentClassifier(ABC):
    
    def __init__(self):
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        self.cache_db = cache_dir / f"{self.__class__.__name__}.db"
        self._init_cache()
    
    def _init_cache(self):
        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                doc_hash TEXT PRIMARY KEY,
                result TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    @abstractmethod
    def _score_single_document(self, document):
        pass
    
    @abstractmethod
    def _score_documents_impl(self, documents):
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
    
    def _load_from_cache(self, document):
        doc_hash = self._get_document_hash(document)
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.execute("SELECT result FROM cache WHERE doc_hash = ?", (doc_hash,))
        row = cursor.fetchone()
        conn.close()
        return json.loads(row[0]) if row else None
    
    def _save_to_cache(self, document, result):
        doc_hash = self._get_document_hash(document)
        conn = sqlite3.connect(self.cache_db)
        conn.execute("INSERT OR REPLACE INTO cache (doc_hash, result) VALUES (?, ?)", 
                    (doc_hash, json.dumps(result)))
        conn.commit()
        conn.close()
    
    def score_documents(self, documents):
        classifier_name = self.__class__.__name__
        console.log(f"[bold cyan]Scoring documents with {classifier_name} (with caching)...[/bold cyan]")
        
        results, docs_to_score = [], []
        cache_hits = cache_misses = 0
        
        for doc in documents:
            cached_result = self._load_from_cache(doc)
            if cached_result is not None:
                results.append(cached_result)
                cache_hits += 1
            else:
                docs_to_score.append(doc)
                cache_misses += 1
        
        console.log(f"[green]Cache hits: {cache_hits}, Cache misses: {cache_misses}[/green]")
        
        if docs_to_score:
            new_results = self._score_documents_impl(docs_to_score)
            for result in new_results:
                doc = next(d for d in docs_to_score if d['id'] == result['id'])
                self._save_to_cache(doc, result)
                results.append(result)
        
        doc_id_to_idx = {doc['id']: idx for idx, doc in enumerate(documents)}
        results.sort(key=lambda r: doc_id_to_idx[r['id']])
        return results

