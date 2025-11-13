import numpy as np
import torch
import random
import yaml
from utils import load_fineweb_documents, load_benchmark_samples, inject_benchmarks_into_documents
from analysis import analyze_and_plot
from rich.console import Console
import models

console = Console()

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main(config_path="config.yaml"):
    config = load_config(config_path)
    set_seed(config["experiment"]["seed"])
    
    console.rule("[bold blue]Haystack Experiment Start[/bold blue]")
    inject_inside = config["experiment"]["inject_inside"]
    num_docs = config["dataset"]["num_docs"]
    
    mmlu_samples = load_benchmark_samples(
        "mmlu", 
        count=config["benchmarks"]["mmlu"]["count"], 
        subjects=config["benchmarks"]["mmlu"]["subjects"]
    )
    gsm8k_samples = load_benchmark_samples("gsm8k", count=config["benchmarks"]["gsm8k"]["count"])
    gpqa_samples = load_benchmark_samples("gpqa", count=config["benchmarks"]["gpqa"]["count"])
    
    num_benchmarks = len(mmlu_samples) + len(gsm8k_samples) + len(gpqa_samples)
    num_fineweb_docs = num_docs if inject_inside else num_docs - num_benchmarks
    
    documents = load_fineweb_documents(
        num_fineweb_docs,
        prefilter_hq=config["dataset"]["prefilter_hq"],
        min_hq_score=config["dataset"]["min_hq_score"],
        fineweb_path=config["dataset"]["fineweb_path"]
    )
    
    benchmark_positions = inject_benchmarks_into_documents(
        documents, mmlu_samples, gsm8k_samples, gpqa_samples, inject_inside=inject_inside
    )
    
    console.log(f"[bold green]Total documents: {len(documents)}[/bold green]")
    
    results = {}
    for clf_config in config["classifiers"]:
        if not clf_config["enabled"]:
            continue
        clf_class = getattr(models, clf_config["name"])
        console.rule(f"[bold blue]Scoring with {clf_config['name']}[/bold blue]")
        clf = clf_class()
        results[clf_config["name"]] = clf.score_documents(documents)
    
    output_base_dir = config.get("output", {}).get("base_dir", "results")
    analyze_and_plot(
        results, 
        documents, 
        benchmark_positions, 
        output_base_dir=output_base_dir, 
        inject_inside=inject_inside,
        prefilter_hq=config["dataset"]["prefilter_hq"],
        num_docs=num_docs
    )
    console.rule("[bold green]Analysis completed.[/bold green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run haystack experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
