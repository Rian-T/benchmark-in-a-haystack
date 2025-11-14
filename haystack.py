from utils import (
    load_fineweb_documents,
    load_benchmark_samples,
    inject_benchmarks_into_documents,
    load_config,
    set_seed,
    get_models_dir
)
from analysis import analyze_and_plot
from rich.console import Console
import models

console = Console()

def download_all_models(config_path="config.yaml"):
    """Download all models specified in the configuration file."""
    config = load_config(config_path)
    models_dir = get_models_dir(config)
    
    console.rule("[bold blue]Model Download Mode[/bold blue]")
    console.log(f"[yellow]Downloading all models to: {models_dir}[/yellow]")
    
    # Get all classifier classes from config
    for clf_config in config["classifiers"]:
        clf_name = clf_config["name"]
        try:
            clf_class = getattr(models, clf_name)
            if hasattr(clf_class, 'download_model'):
                console.rule(f"[bold cyan]Downloading {clf_name}[/bold cyan]")
                clf_class.download_model(models_dir=models_dir)
            else:
                console.log(f"[yellow]Warning: {clf_name} does not have a download_model method[/yellow]")
        except AttributeError:
            console.log(f"[red]Error: Classifier {clf_name} not found in models module[/red]")
        except Exception as e:
            console.log(f"[red]Error downloading {clf_name}: {e}[/red]")
    
    console.rule("[bold green]All models downloaded successfully![/bold green]")

def main(config_path="config.yaml"):
    config = load_config(config_path)
    set_seed(config["experiment"]["seed"])
    
    console.rule("[bold blue]Haystack Experiment Start[/bold blue]")
    inject_inside = config["experiment"]["inject_inside"]
    num_docs = config["dataset"]["num_docs"]
    
    # Dynamically load all benchmarks from config
    benchmark_samples_dict = {}
    total_benchmark_count = 0
    
    for benchmark_type, benchmark_config in config["benchmarks"].items():
        # Extract count and subjects (if present)
        count = benchmark_config.get("count", 5)
        subjects = benchmark_config.get("subjects", None)
        
        console.log(f"[cyan]Loading benchmark: {benchmark_type} (count={count})[/cyan]")
        samples = load_benchmark_samples(benchmark_type, count=count, subjects=subjects)
        benchmark_samples_dict[benchmark_type] = samples
        total_benchmark_count += len(samples)
    
    console.log(f"[bold green]Loaded {len(benchmark_samples_dict)} benchmark types with {total_benchmark_count} total samples[/bold green]")
    
    num_fineweb_docs = num_docs if inject_inside else num_docs - total_benchmark_count
    
    documents = load_fineweb_documents(
        num_fineweb_docs,
        prefilter_hq=config["dataset"]["prefilter_hq"],
        min_hq_score=config["dataset"]["min_hq_score"],
        fineweb_path=config["dataset"]["fineweb_path"]
    )
    
    benchmark_positions = inject_benchmarks_into_documents(
        documents, benchmark_samples_dict, inject_inside=inject_inside
    )
    
    console.log(f"[bold green]Total documents: {len(documents)}[/bold green]")
    
    # Add models_dir to classifier configs
    models_dir = get_models_dir(config)
    
    # Extract dataset name from fineweb_path for cache organization
    fineweb_path = config["dataset"]["fineweb_path"]
    dataset_name = fineweb_path.split("/")[-1] if "/" in fineweb_path else fineweb_path
    console.log(f"[cyan]Using dataset: {dataset_name}[/cyan]")
    
    results = {}
    for clf_config in config["classifiers"]:
        if not clf_config["enabled"]:
            continue
        # Pass models_dir and dataset_name to classifier config
        clf_config_with_models = clf_config.copy()
        clf_config_with_models["models_dir"] = models_dir
        clf_config_with_models["dataset_name"] = dataset_name
        
        clf_class = getattr(models, clf_config["name"])
        console.rule(f"[bold blue]Scoring with {clf_config['name']}[/bold blue]")
        clf = clf_class(clf_config_with_models)
        results[clf_config["name"]] = clf.score_documents(documents)
    
    output_base_dir = config.get("output", {}).get("base_dir", "results")
    analyze_and_plot(
        results, 
        documents, 
        benchmark_positions, 
        output_base_dir=output_base_dir, 
        inject_inside=inject_inside,
        prefilter_hq=config["dataset"]["prefilter_hq"],
        num_docs=num_docs,
        dataset_name=dataset_name
    )
    console.rule("[bold green]Analysis completed.[/bold green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run haystack experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--download-models", action="store_true", help="Download all models and exit without running experiment")
    args = parser.parse_args()
    
    if args.download_models:
        download_all_models(args.config)
    else:
        main(args.config)
