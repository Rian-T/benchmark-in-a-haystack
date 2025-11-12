import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
import random
import pandas as pd
import os
import json
import fasttext
import re
from typing import List
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from benchmarks import BENCHMARKS
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

def load_fineweb_documents(num_docs=100000, prefilter_hq=False, min_hq_score=0.5, fineweb_path="HuggingFaceFW/fineweb"):
    """Load documents from the fineweb dataset."""
    console.rule("[bold blue]Loading fineweb dataset...[/bold blue]")
    fineweb = load_dataset(fineweb_path, name="sample-10BT", split="train", streaming=True)
    
    documents = []
    
    if prefilter_hq:
        console.log(f"[yellow]Pre-filtering documents for high quality (min score: {min_hq_score})...[/yellow]")
        console.log(f"Will continue loading until {num_docs} high-quality documents are found...")
        model = load_fasttext_model()
        counter = 0
        processed_docs = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Finding high-quality documents...", total=num_docs)
            
            for doc in fineweb:
                counter += 1
                processed_docs += 1
                
                text = doc["text"].replace("\n", " ")
                labels, probs = model.predict(text, k=2)
                
                hq_prob = 0.0
                for j, label in enumerate(labels):
                    if label == "__label__hq":
                        hq_prob = probs[j]
                        break
                
                if hq_prob >= min_hq_score:
                    documents.append({
                        "id": f"fineweb_{len(documents)}",
                        "source": "fineweb",
                        "text": doc["text"],
                        "contains_benchmark": False,
                        "benchmark_type": None,
                        "original_text": doc["text"],
                        "original_score": float(hq_prob)
                    })
                    progress.update(task, advance=1)
                
                if len(documents) >= num_docs:
                    break
                
        console.log(f"[green]Found {len(documents)} high-quality documents after processing {processed_docs} documents ({len(documents)/processed_docs:.2%} acceptance rate)[/green]")
    else:
        console.log(f"[yellow]Collecting {num_docs} documents without quality filtering...[/yellow]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[green]Loading documents...", total=num_docs)
            for i, doc in enumerate(fineweb.take(num_docs)):
                documents.append({
                    "id": f"fineweb_{i}",
                    "source": "fineweb",
                    "text": doc["text"],
                    "contains_benchmark": False,
                    "benchmark_type": None,
                    "original_text": doc["text"]
                })
                progress.update(task, advance=1)
    
    console.log(f"[bold green]Loaded {len(documents)} documents[/bold green]")
    return documents

def load_benchmark_samples(benchmark_type, count=5, subjects=None):
    """Load benchmark samples using the Benchmark class interface."""
    console.rule(f"[bold blue]Loading {benchmark_type} dataset...[/bold blue]")
    if benchmark_type not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    benchmark = BENCHMARKS[benchmark_type]
    samples = benchmark.load_samples(count=count, subjects=subjects)
    console.log(f"[green]Loaded {len(samples)} {benchmark_type} samples[/green]")
    return samples

def format_benchmark_text(sample, benchmark_type, subject=None):
    """Format a benchmark sample as text using the Benchmark class interface."""
    if benchmark_type not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark type: {benchmark_type}")
    benchmark = BENCHMARKS[benchmark_type]
    return benchmark.format_sample(sample, subject=subject)

def inject_benchmarks_into_documents(documents, mmlu_samples, gsm8k_samples=None, gpqa_samples=None, inject_inside=True):
    """Add benchmark samples either by injecting them or creating separate documents."""
    console.rule(f"[bold blue]Adding benchmark samples as {'injected content' if inject_inside else 'separate documents'}...[/bold blue]")
    benchmark_positions = []
    
    num_docs = len(documents)
    ranges = {
        "mmlu": (0, int(0.4*num_docs)),
        "gsm8k": (int(0.5*num_docs), int(0.9*num_docs)),
        "gpqa": (int(0.4*num_docs), int(0.5*num_docs))
    }
    
    all_samples = []
    
    for i, sample in enumerate(mmlu_samples):
        all_samples.append({
            "sample": sample,
            "benchmark_type": "mmlu",
            "index": i,
            "subject": sample.get("subject", None)
        })
    
    if gsm8k_samples:
        for i, sample in enumerate(gsm8k_samples):
            all_samples.append({
                "sample": sample,
                "benchmark_type": "gsm8k",
                "index": i,
                "subject": None
            })
    
    if gpqa_samples:
        for i, sample in enumerate(gpqa_samples):
            all_samples.append({
                "sample": sample,
                "benchmark_type": "gpqa",
                "index": i,
                "subject": None
            })
    
    for benchmark in all_samples:
        benchmark_type = benchmark["benchmark_type"]
        index = benchmark["index"]
        sample = benchmark["sample"]
        subject = benchmark.get("subject")
        
        benchmark_text = format_benchmark_text(sample, benchmark_type, subject)
        
        if inject_inside:
            range_min, range_max = ranges[benchmark_type]
            doc_index = random.randint(range_min, min(range_max, len(documents)-1))
            
            if len(documents[doc_index]['text']) > 5000:
                split_point = len(documents[doc_index]['text']) // 2
                documents[doc_index]['text'] = (
                    documents[doc_index]['text'][:split_point] +
                    "\n\n" + benchmark_text + "\n\n" +
                    documents[doc_index]['text'][split_point:]
                )
            else:
                documents[doc_index]['text'] += "\n\n" + benchmark_text
            
            documents[doc_index]['contains_benchmark'] = True
            documents[doc_index]['benchmark_type'] = benchmark_type
            documents[doc_index]['benchmark_index'] = index
            if subject:
                documents[doc_index]['benchmark_subject'] = subject
            
            benchmark_positions.append({
                "doc_id": documents[doc_index]['id'],
                "doc_index": doc_index,
                "benchmark_type": benchmark_type,
                "index": index,
                "subject": subject
            })
            
            console.log(f"[cyan]Injected {benchmark_type} sample {index} into document {documents[doc_index]['id']}[/cyan]")
        else:
            new_doc = {
                "id": f"{benchmark_type}_{index}",
                "source": benchmark_type,
                "text": benchmark_text,
                "contains_benchmark": True,
                "benchmark_type": benchmark_type,
                "benchmark_index": index,
                "original_text": benchmark_text
            }
            
            if subject:
                new_doc["benchmark_subject"] = subject
            
            doc_index = len(documents)
            documents.append(new_doc)
            
            benchmark_positions.append({
                "doc_id": new_doc['id'],
                "doc_index": doc_index,
                "benchmark_type": benchmark_type,
                "index": index,
                "subject": subject
            })
            
            console.log(f"[cyan]Created new document for {benchmark_type} sample {index}[/cyan]")
    
    return benchmark_positions

def load_fasttext_model(model_path="models/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"):
    """Load the fasttext model from the specified file path."""
    console.log(f"[yellow]Loading fasttext model from {model_path}...[/yellow]")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FastText model file not found at: {model_path}")
    return fasttext.load_model(model_path)

def score_documents(documents, model):
    """Score all documents with the fasttext model."""
    console.rule("[bold blue]Scoring documents...[/bold blue]")
    scores = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Scoring documents...", total=len(documents))
        for doc in documents:
            try:
                text = doc["text"].replace("\n", " ")
                labels, probs = model.predict(text, k=2)
                hq_prob = next((probs[i] for i, label in enumerate(labels) if label == "__label__hq"), 0.0)
                scores.append({
                    "id": doc["id"],
                    "source": doc["source"],
                    "contains_benchmark": doc["contains_benchmark"],
                    "benchmark_type": doc["benchmark_type"],
                    "benchmark_index": doc.get("benchmark_index", None),
                    "score": float(hq_prob)
                })
            except Exception as e:
                console.log(f"[red]Error processing document {doc['id']}: {e}[/red]")
                scores.append({
                    "id": doc["id"],
                    "source": doc["source"],
                    "contains_benchmark": doc["contains_benchmark"],
                    "benchmark_type": doc["benchmark_type"],
                    "benchmark_index": doc.get("benchmark_index", None),
                    "score": None
                })
            progress.update(task, advance=1)
    return scores

def analyze_scores(scores, documents, benchmark_positions, inject_inside=True, prefilter_hq=False, prefix=""):
    """Analyze and report score statistics."""
    console.rule("[bold blue]Analyzing scores...[/bold blue]")
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.dropna(subset=["score"])
    scores_df = scores_df.sort_values("score", ascending=False)
    scores_df["rank"] = range(1, len(scores_df) + 1)
    scores_df.to_csv(f"{prefix}haystack_scores.csv", index=False)
    
    benchmark_ranks = scores_df[scores_df["contains_benchmark"] == True]
    total_docs = len(scores_df)
    benchmark_results = []
    
    for _, row in benchmark_ranks.iterrows():
        percentile = (total_docs - row["rank"]) / total_docs * 100
        
        benchmark_type = row["benchmark_type"]
        benchmark_index = row["benchmark_index"]
        console.log(f"[magenta]Benchmark {benchmark_type} sample {benchmark_index} (in document {row['id']}) ranked {row['rank']}/{total_docs} (top {percentile:.2f}%) with score {row['score']:.4f}[/magenta]")
        
        result = {
            "id": row["id"],
            "rank": int(row["rank"]),
            "total_docs": total_docs,
            "percentile": float(percentile),
            "score": float(row["score"])
        }
        benchmark_results.append(result)
    
    with open(f"{prefix}benchmark_rankings_{'injected' if inject_inside else 'separate'}.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    console.log(f"[bold green]Mean score: {scores_df['score'].mean():.4f}[/bold green]")
    console.log(f"[bold green]Median score: {scores_df['score'].median():.4f}[/bold green]")
    console.log(f"[bold green]Min score: {scores_df['score'].min():.4f}[/bold green]")
    console.log(f"[bold green]Max score: {scores_df['score'].max():.4f}[/bold green]")
    
    percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    percentile_results = {}
    
    for p in percentiles:
        threshold = np.percentile(scores_df["score"], 100 - p)
        percentile_results[str(p)] = float(threshold)
        console.log(f"[cyan]Top {p}% threshold: {threshold:.4f}[/cyan]")
    
    with open(f"{prefix}score_thresholds.json", "w") as f:
        json.dump(percentile_results, f, indent=2)
    
    return scores_df, benchmark_ranks

def analyze_benchmark_effect(documents, benchmark_positions, benchmark_ranks, model, inject_inside=True, prefilter_hq=False, prefix=""):
    """Analyze the effect of benchmark injection on document scores."""
    console.rule("[bold blue]Benchmark Effect Analysis...[/bold blue]")
    results = []
    
    for i, pos in enumerate(benchmark_positions):
        doc_index = pos["doc_index"]
        doc = documents[doc_index]
        
        if doc["source"] in ["mmlu", "gsm8k", "gpqa"]:
            benchmark_type = doc["benchmark_type"]
            benchmark_index = doc["benchmark_index"]
            benchmark_score = float(benchmark_ranks[benchmark_ranks["id"] == doc["id"]].iloc[0]["score"])
            
            results.append({
                "doc_id": doc["id"],
                "subject": pos.get("subject", None),
                "is_standalone": True,
                "original_score": None,
                "benchmark_score": benchmark_score,
                "difference": None
            })
            continue
        
        try:
            original_text = doc["original_text"].replace("\n", " ")
            labels, probs = model.predict(original_text, k=2)
            
            orig_hq_prob = 0.0
            for j, label in enumerate(labels):
                if label == "__label__hq":
                    orig_hq_prob = probs[j]
                    break
            
            benchmark_doc = benchmark_ranks[benchmark_ranks["id"] == doc["id"]]
            if not benchmark_doc.empty:
                benchmark_score = benchmark_doc.iloc[0]["score"]
                console.log(f"[magenta]Document {doc['id']} - Original score: {orig_hq_prob:.4f}, With benchmark: {benchmark_score:.4f}, Difference: {benchmark_score - orig_hq_prob:.4f}[/magenta]")
                
                results.append({
                    "doc_id": doc["id"],
                    "subject": pos.get("subject", None),
                    "is_standalone": False,
                    "original_score": float(orig_hq_prob),
                    "benchmark_score": float(benchmark_score),
                    "difference": float(benchmark_score - orig_hq_prob)
                })
        except Exception as e:
            console.log(f"[red]Error analyzing original document {doc['id']}: {e}[/red]")
    
    with open(f"{prefix}benchmark_effect_{'injected' if inject_inside else 'separate'}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results
