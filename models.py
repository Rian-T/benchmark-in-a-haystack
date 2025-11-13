import os
import re
import shutil
import torch
import fasttext
from abc import abstractmethod
from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from tqdm import tqdm
from utils import DocumentClassifier, score_documents, load_fasttext_model


console = Console()

class DCLMClassifier(DocumentClassifier):
    def __init__(self):
        super().__init__()
        console.log("[bold cyan]Initializing DCLMClassifier...[/bold cyan]")
        self.model = self._load_model()

    @staticmethod
    def _load_model():
        model_path = "models/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
        if not os.path.exists(model_path):
            console.log(f"[yellow]Model not found at {model_path}. Downloading...[/yellow]")
            os.makedirs("models", exist_ok=True)
            downloaded_path = hf_hub_download(
                "mlfoundations/fasttext-oh-eli5",
                "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
            )
            shutil.copy(downloaded_path, model_path)
            console.log(f"[green]Model downloaded to {model_path}.[/green]")
        return load_fasttext_model(model_path)

    def _score_single_document(self, document):
        pass

    def _score_documents_impl(self, documents):
        console.log("[bold cyan]Scoring documents with DCLMClassifier...[/bold cyan]")
        return score_documents(documents, self.model)

class TextbookFastTextClassifier(DocumentClassifier):
    def __init__(self):
        super().__init__()
        console.log("[bold cyan]Initializing TextbookFastTextClassifier...[/bold cyan]")
        self.model = self._load_model()

    @staticmethod
    def _load_model():
        model_path = "models/textbook_model.bin"
        if os.path.exists(model_path):
            console.log(f"[yellow]Loading Textbook FastText model from local {model_path}...[/yellow]")
            return fasttext.load_model(model_path)
        else:
            console.log("[yellow]Loading Textbook FastText model from HuggingFace Hub...[/yellow]")
            return fasttext.load_model(
                hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifer-v1", "model.bin")
            )

    def _score_single_document(self, document):
        pass

    def _score_documents_impl(self, documents):
        console.log("[bold cyan]Scoring documents with TextbookFastTextClassifier...[/bold cyan]")
        texts = [re.sub(r"\n+", " ", doc["text"]) for doc in documents]
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

class TransformerClassifier(DocumentClassifier):
    
    def __init__(self):
        super().__init__()
        console.log(f"[bold cyan]Initializing {self.__class__.__name__}...[/bold cyan]")
        config = self.get_model_config()
        self.tokenizer, self.model, self.device = self._load_transformer_model(
            config['model_dir'], 
            config['hub_name'], 
            config.get('trust_remote_code', False),
            config.get('torch_dtype')
        )
        self.batch_size = 100

    @abstractmethod
    def get_model_config(self):
        pass

    @abstractmethod
    def process_outputs(self, outputs, doc_batch):
        pass

    def _score_single_document(self, document):
        pass

    def _score_documents_impl(self, documents):
        console.log(f"[bold cyan]Scoring documents with {self.__class__.__name__}...[/bold cyan]")
        results = []
        for idx_batch in tqdm(range(0, len(documents), self.batch_size)):
            doc_batch = documents[idx_batch:idx_batch + self.batch_size]
            text_batch = [doc["text"] for doc in doc_batch]
            
            config = self.get_model_config()
            tokenizer_kwargs = {"return_tensors": "pt", "padding": "longest", "truncation": True}
            if config.get('max_length'):
                tokenizer_kwargs["max_length"] = config['max_length']
            
            inputs = self.tokenizer(text_batch, **tokenizer_kwargs).to(self.device)
            inputs = self._process_inputs(inputs)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results.extend(self.process_outputs(outputs, doc_batch))
        
        return results

    def _process_inputs(self, inputs):
        return inputs


class FinewebEduClassifier(TransformerClassifier):
    
    def get_model_config(self):
        return {
            'model_dir': "models/fineweb-edu-classifier",
            'hub_name': "HuggingFaceTB/fineweb-edu-classifier",
            'trust_remote_code': False
        }
    
    def process_outputs(self, outputs, doc_batch):
        results = []
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


class GaperonClassifier(TransformerClassifier):

    def get_model_config(self):
        return {
            'model_dir': "models/gaperon-quality-cls",
            'hub_name': "almanach/gaperon-quality-cls",
            'trust_remote_code': True,
            'max_length': 512
        }
    
    def _process_inputs(self, inputs):
        return {k: v[:, :512] for k, v in inputs.items()}
    
    def process_outputs(self, outputs, doc_batch):
        results = []
        for i_doc, doc in enumerate(doc_batch):
            logits = outputs.logits_list[0][i_doc].squeeze(0).float().softmax(-1).detach().cpu().numpy()
            score = (logits[0] + 0.5 * logits[2]).item()
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


class NemoCuratorEduClassifier(TransformerClassifier):

    def get_model_config(self):
        return {
            'model_dir': "models/nemocurator-fineweb-mixtral-edu-classifier",
            'hub_name': "nvidia/nemocurator-fineweb-mixtral-edu-classifier",
            'trust_remote_code': False,
            'max_length': 512,
            'torch_dtype': torch.bfloat16
        }
    
    def process_outputs(self, outputs, doc_batch):
        results = []
        for i_doc, doc in enumerate(doc_batch):
            logit = outputs.logits[i_doc].squeeze(-1).float().cpu().numpy()
            score = float(logit)
            int_score = int(round(max(0, min(score, 5))))
            pred_label = "high_quality" if score >= 2.5 else "low_quality"
            results.append({
                "id": doc["id"],
                "source": doc["source"],
                "contains_benchmark": doc["contains_benchmark"],
                "benchmark_type": doc["benchmark_type"],
                "benchmark_index": doc.get("benchmark_index", None),
                "score": score,
                "int_score": int_score,
                "label": pred_label
            })
        return results


class FinePDFsClassifierBase(DocumentClassifier):
    
    def __init__(self):
        super().__init__()
        console.log(f"[bold cyan]Initializing {self.__class__.__name__}...[/bold cyan]")
        config = self.get_model_config()
        self.tokenizer, self.model, self.device = self._load_transformer_model(
            config['model_dir'], config['hub_name']
        )
        self.CHUNK_SIZE = 2046
        self.MAX_CHARS = 10_000
    
    @abstractmethod
    def get_model_config(self):
        pass
    
    def _trim_to_whitespace(self, text, trim_start, trim_end):
        if trim_start:
            match = re.search(r'\s', text)
            text = text[match.start()+1:] if match else text[10:]
        if trim_end:
            match = re.search(r'\s', text[::-1])
            text = text[:len(text) - match.start() - 1] if match else text[:-10]
        return text
    
    def _create_text_chunks(self, text):
        if len(text) <= 2 * self.MAX_CHARS:
            tokens = self.tokenizer.encode(text[:self.MAX_CHARS], return_tensors="np", add_special_tokens=False)[0]
            chunk_text = self.tokenizer.decode(tokens[:self.CHUNK_SIZE], skip_special_tokens=True)
            return [self._trim_to_whitespace(chunk_text, False, True)]
        
        text_top, text_bottom = text[:self.MAX_CHARS], text[-self.MAX_CHARS:]
        tokens = self.tokenizer.batch_encode_plus([text_top, text_bottom], return_tensors="np", add_special_tokens=False)["input_ids"]
        chunks = [tokens[0][:self.CHUNK_SIZE], tokens[1][-self.CHUNK_SIZE:]]
        chunks_text = self.tokenizer.batch_decode(chunks, skip_special_tokens=True)
        return [
            self._trim_to_whitespace(chunks_text[0], False, True),
            self._trim_to_whitespace(chunks_text[1], True, False)
        ]
    
    def _score_single_document(self, document):
        pass
    
    def _score_documents_impl(self, documents):
        console.log(f"[bold cyan]Scoring documents with {self.__class__.__name__}...[/bold cyan]")
        results = []
        
        for doc in tqdm(documents):
            scores = []
            for chunk in self._create_text_chunks(doc["text"]):
                inputs = self.tokenizer(chunk, return_tensors="pt", padding="longest", truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                scores.append(outputs.logits.squeeze(-1).float().detach().cpu().numpy().item())
            
            final_score = max(scores)
            results.append({
                "id": doc["id"],
                "source": doc["source"],
                "contains_benchmark": doc["contains_benchmark"],
                "benchmark_type": doc["benchmark_type"],
                "benchmark_index": doc.get("benchmark_index", None),
                "score": float(final_score),
                "int_score": int(round(max(0, min(final_score, 5))))
            })
        
        return results


class FinePDFsEduClassifier(FinePDFsClassifierBase):
    
    def get_model_config(self):
        return {
            'model_dir': "models/finepdfs-edu-classifier-eng-Latn",
            'hub_name': "HuggingFaceFW/finepdfs_edu_classifier_eng_Latn"
        }


class FinePDFsEduClassifierV2(FinePDFsClassifierBase):
    
    def get_model_config(self):
        return {
            'model_dir': "models/finepdfs-edu-classifier-v2-eng-Latn",
            'hub_name': "HuggingFaceFW/finepdfs_edu_classifier_v2_eng_Latn"
        }


class FinePDFsDCLMClassifier(FinePDFsClassifierBase):
    
    def get_model_config(self):
        return {
            'model_dir': "models/finepdfs-dclm-classifier-eng-Latn",
            'hub_name': "HuggingFaceFW/finepdfs_dclm_classifier_eng_Latn"
        }


class EuroFilterClassifier(TransformerClassifier):

    def get_model_config(self):
        return {
            'model_dir': "models/eurofilter-v1",
            'hub_name': "utter-project/EuroFilter-v1",
            'trust_remote_code': True,
            'max_length': 512,
            'torch_dtype': torch.bfloat16
        }
    
    def process_outputs(self, outputs, doc_batch):
        results = []
        for i_doc, doc in enumerate(doc_batch):
            score = outputs.logits[i_doc].squeeze().float().cpu().numpy().item()
            score = max(0, min(score, 5))
            int_score = int(round(score))
            
            prob = torch.nn.functional.sigmoid(outputs.binary_logits[i_doc]).cpu().numpy().item()
            binary_pred = 1 if prob >= 0.5 else 0
            
            results.append({
                "id": doc["id"],
                "source": doc["source"],
                "contains_benchmark": doc["contains_benchmark"],
                "benchmark_type": doc["benchmark_type"],
                "benchmark_index": doc.get("benchmark_index", None),
                "score": float(score),
                "int_score": int_score,
                "binary_pred": binary_pred,
                "prob": float(prob)
            })
        return results
