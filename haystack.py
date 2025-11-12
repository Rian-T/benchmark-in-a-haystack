import numpy as np
import torch
import random
import os
import requests
from utils import (
    load_fineweb_documents,
    load_benchmark_samples,
    inject_benchmarks_into_documents,
    score_documents
)
from analysis import analyze_and_plot
from abc import ABC, abstractmethod
from models import DCLMClassifier, TextbookFastTextClassifier, FinewebEduClassifier, GaperonClassifier, FinePDFsEduClassifier, FinePDFsDCLMClassifier
from rich.console import Console
from huggingface_hub import hf_hub_download

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

console = Console()

CLASSIFIERS = [GaperonClassifier, FinewebEduClassifier, DCLMClassifier, TextbookFastTextClassifier, FinePDFsEduClassifier, FinePDFsDCLMClassifier]

def download_all_models():
    """Download all required models to the local 'models' folder."""
    from huggingface_hub import hf_hub_download
    import shutil

    console.rule("[bold blue]Downloading all required models...[/bold blue]")

    dclm_path = "models/openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
    if not os.path.exists(dclm_path):
        bin_path = hf_hub_download("mlfoundations/fasttext-oh-eli5", "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin")
        os.makedirs("models", exist_ok=True)
        shutil.copy(bin_path, dclm_path)
        console.log(f"[green]Downloaded DCLM FastText model.[/green]")
    else:
        console.log(f"[green]DCLM FastText model already exists at {dclm_path}.[/green]")

    tb_path = "models/textbook_model.bin"
    if not os.path.exists(tb_path):
        console.log(f"[yellow]Downloading Textbook FastText model to {tb_path}...[/yellow]")
        bin_path = hf_hub_download("kenhktsui/llm-data-textbook-quality-fasttext-classifer-v1", "model.bin")
        os.makedirs("models", exist_ok=True)
        shutil.copy(bin_path, tb_path)
        console.log(f"[green]Downloaded Textbook FastText model.[/green]")
    else:
        console.log(f"[green]Textbook FastText model already exists at {tb_path}.[/green]")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_dir = "models/fineweb-edu-classifier"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    
    try:
        console.log(f"[yellow]Downloading FinewebEduClassifier model and tokenizer to {model_dir}...[/yellow]")
        AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier").save_pretrained(model_dir)
        AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier").save_pretrained(model_dir)
        console.log(f"[green]Downloaded FinewebEduClassifier model and tokenizer.[/green]")
    except Exception as e:
        console.log(f"[red]Error downloading FinewebEduClassifier: {e}[/red]")
    
    model_dir = "models/gaperon-classifier"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    
    try:
        console.log(f"[yellow]Downloading GaperonClassifier model and tokenizer to {model_dir}...[/yellow]")
        AutoTokenizer.from_pretrained("almanach/gaperon-quality-cls").save_pretrained(model_dir)
        AutoModelForSequenceClassification.from_pretrained(
            "almanach/gaperon-quality-cls", trust_remote_code=True
        ).save_pretrained(model_dir)
        console.log(f"[green]Downloaded GaperonClassifier model and tokenizer.[/green]")
    except Exception as e:
        console.log(f"[red]Error downloading GaperonClassifier: {e}[/red]")

    console.rule("[bold green]All models downloaded.[/bold green]")

def main(inject_inside=True, num_docs=100000, prefilter_hq=False, min_hq_score=0.5, fineweb_path="HuggingFaceFW/fineweb", mmlu_count=3, mmlu_subjects=None):
    console.rule("[bold blue]Haystack Experiment Start[/bold blue]")
    console.log(f"[bold green]Running experiment with {'injected' if inject_inside else 'separate'} benchmarks on "
                f"{'pre-filtered high-quality' if prefilter_hq else 'unfiltered'} documents[/bold green]")

    console.log("[yellow]Loading benchmark samples...[/yellow]")
    if mmlu_subjects is None:
        mmlu_subjects = ["anatomy", "computer_security", "high_school_geography", "moral_scenarios", "college_physics"]
    mmlu_samples = load_benchmark_samples("mmlu", count=mmlu_count, subjects=mmlu_subjects)
    gsm8k_samples = load_benchmark_samples("gsm8k", count=10)
    gpqa_samples = load_benchmark_samples("gpqa", count=10)
    all_benchmarks = mmlu_samples + gsm8k_samples + gpqa_samples
    num_benchmarks = len(all_benchmarks)

    if inject_inside:
        num_fineweb_docs = num_docs
    else:
        num_fineweb_docs = num_docs - num_benchmarks
        if num_fineweb_docs < 1:
            raise ValueError("Number of documents too small for the number of benchmarks.")

    documents = load_fineweb_documents(
        num_fineweb_docs,
        prefilter_hq=prefilter_hq,
        min_hq_score=min_hq_score,
        fineweb_path=fineweb_path
    )

    benchmark_positions = inject_benchmarks_into_documents(
        documents, mmlu_samples, gsm8k_samples, gpqa_samples, inject_inside=inject_inside
    )

    assert len(documents) == num_docs, f"Final document count {len(documents)} != requested {num_docs}"
    console.log(f"[bold green]Total documents after benchmark injection: {len(documents)}[/bold green]")

    results = {}
    for clf_class in CLASSIFIERS:
        console.rule(f"[bold blue]Scoring with {clf_class.__name__}[/bold blue]")
        clf = clf_class()
        results[clf.__class__.__name__] = clf.score_documents(documents)

    console.rule("[bold blue]Analyzing and plotting results...[/bold blue]")
    analyze_and_plot(results, documents, benchmark_positions)

    console.rule("[bold green]Analysis completed. Results saved to CSV/JSON and plot.[/bold green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run haystack experiment")
    parser.add_argument("--separate", action="store_true", help="Create separate documents for benchmarks")
    parser.add_argument("--num-docs", type=int, default=100000, help="Number of documents to load")
    parser.add_argument("--prefilter-hq", action="store_true", help="Pre-filter documents for high quality")
    parser.add_argument("--min-hq-score", type=float, default=0.7, help="Minimum high-quality score threshold")
    parser.add_argument("--download-models", action="store_true", help="Download all required models and exit")
    parser.add_argument("--fineweb-path", type=str, default="HuggingFaceFW/fineweb", help="Path or HF repo for fineweb dataset")
    parser.add_argument("--mmlu-count", type=int, default=3, help="Number of MMLU samples per subject (default: 3)")
    parser.add_argument("--mmlu-subjects", type=str, default=None, help="Comma-separated list of MMLU subjects (default: anatomy,computer_security,high_school_geography,moral_scenarios,college_physics)")
    args = parser.parse_args()

    if args.download_models:
        download_all_models()
        exit(0)

    mmlu_subjects = None
    if args.mmlu_subjects:
        mmlu_subjects = [s.strip() for s in args.mmlu_subjects.split(",")]

    main(
        inject_inside=not args.separate,
        num_docs=args.num_docs,
        prefilter_hq=args.prefilter_hq,
        min_hq_score=args.min_hq_score,
        fineweb_path=args.fineweb_path,
        mmlu_count=args.mmlu_count,
        mmlu_subjects=mmlu_subjects
    )
