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
from models import DCLMClassifier, TextbookFastTextClassifier, FinewebEduClassifier
from rich.console import Console

# Set a random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

console = Console()

CLASSIFIERS = [DCLMClassifier, TextbookFastTextClassifier, FinewebEduClassifier]

def main(inject_inside=True, num_docs=100000, prefilter_hq=False, min_hq_score=0.5):
    console.rule("[bold blue]Haystack Experiment Start[/bold blue]")
    console.log(f"[bold green]Running experiment with {'injected' if inject_inside else 'separate'} benchmarks on "
                f"{'pre-filtered high-quality' if prefilter_hq else 'unfiltered'} documents[/bold green]")

    # 1. Load benchmark samples first to know how many there are
    console.log("[yellow]Loading benchmark samples...[/yellow]")
    mmlu_samples = load_benchmark_samples("mmlu", subjects=["anatomy", "computer_security", "college_physics"])
    gsm8k_samples = load_benchmark_samples("gsm8k", count=5)
    gpqa_samples = load_benchmark_samples("gpqa", count=5)
    all_benchmarks = mmlu_samples + gsm8k_samples + gpqa_samples
    num_benchmarks = len(all_benchmarks)

    # 2. Compute number of fineweb docs to load
    if inject_inside:
        num_fineweb_docs = num_docs
    else:
        num_fineweb_docs = num_docs - num_benchmarks
        if num_fineweb_docs < 1:
            raise ValueError("Number of documents too small for the number of benchmarks.")

    # 3. Load documents
    documents = load_fineweb_documents(num_fineweb_docs, prefilter_hq=prefilter_hq, min_hq_score=min_hq_score)

    # 4. Inject benchmarks
    benchmark_positions = inject_benchmarks_into_documents(
        documents, mmlu_samples, gsm8k_samples, gpqa_samples, inject_inside=inject_inside
    )

    # 5. Check final doc count
    assert len(documents) == num_docs, f"Final document count {len(documents)} != requested {num_docs}"
    console.log(f"[bold green]Total documents after benchmark injection: {len(documents)}[/bold green]")

    # 6. Score documents with all classifiers
    results = {}
    for clf_class in CLASSIFIERS:
        console.rule(f"[bold blue]Scoring with {clf_class.__name__}[/bold blue]")
        clf = clf_class()
        results[clf.__class__.__name__] = clf.score_documents(documents)

    # 7. Analyze and plot results
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
    args = parser.parse_args()

    main(
        inject_inside=not args.separate,
        num_docs=args.num_docs,
        prefilter_hq=args.prefilter_hq,
        min_hq_score=args.min_hq_score
    )
