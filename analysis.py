import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from pathlib import Path
from datetime import datetime

from rich.console import Console

console = Console()

# Set style for beautiful plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 18

def analyze_and_plot(results, documents, benchmark_positions, output_base_dir="results", inject_inside=True, prefilter_hq=False, num_docs=100000, dataset_name="fineweb"):
    """Output benchmark sample ranks across classifiers and create visualizations."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(output_base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    mode_suffix = "injected" if inject_inside else "separate"
    prefilter_suffix = "_prefiltered" if prefilter_hq else ""
    file_suffix = f"_{mode_suffix}{prefilter_suffix}_{num_docs}docs"

    all_benchmark_ranks = []
    plot_data = []
    bench_ranks_dict = {}

    console.rule("[bold blue]Analyzing classifier results...[/bold blue]")

    for clf_name, scores in results.items():
        console.log(f"[yellow]Analyzing results for {clf_name}...[/yellow]")
        scores_df = pd.DataFrame(scores)
        scores_df = scores_df.dropna(subset=["score"])
        scores_df = scores_df.sort_values("score", ascending=False)
        scores_df["rank"] = range(1, len(scores_df) + 1)

        bench_df = scores_df[scores_df["contains_benchmark"] == True].copy()
        bench_df["classifier"] = clf_name
        bench_df["percentile"] = (len(scores_df) - bench_df["rank"]) / len(scores_df) * 100

        for _, row in bench_df.iterrows():
            key = (row["id"], row["benchmark_type"], row["benchmark_index"])
            if key not in bench_ranks_dict:
                bench_ranks_dict[key] = {
                    "id": row["id"],
                    "benchmark_type": row["benchmark_type"],
                    "benchmark_index": row["benchmark_index"],
                }
            bench_ranks_dict[key][clf_name] = {
                "rank": int(row["rank"]),
                "percentile": float(row["percentile"]),
                "score": float(row["score"])
            }

        all_benchmark_ranks.append(bench_df)
        plot_data.append(bench_df[["classifier", "benchmark_type", "rank", "percentile"]])

    bench_ranks_json = os.path.join(results_dir, f"benchmark_ranks_all_classifiers{file_suffix}.json")
    with open(bench_ranks_json, "w") as f:
        json.dump(list(bench_ranks_dict.values()), f, indent=2)
    console.log(f"[green]Saved all benchmark ranks to {bench_ranks_json}[/green]")

    plot_rows = []
    for bench in bench_ranks_dict.values():
        for clf_name in results.keys():
            if clf_name in bench:
                plot_rows.append({
                    "benchmark_id": bench["id"],
                    "benchmark_type": bench["benchmark_type"],
                    "classifier": clf_name,
                    "rank": bench[clf_name]["rank"],
                    "percentile": bench[clf_name]["percentile"],
                    "score": bench[clf_name]["score"]
                })
    plot_df = pd.DataFrame(plot_rows)

    console.log("[yellow]Plotting benchmark sample ranks by classifier and benchmark type...[/yellow]")
    num_classifiers = len(results)
    fig_width = max(16, num_classifiers * 2.5)  # More width for better spacing
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(fig_width, 11), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Use standard, easily distinguishable colors
    # Using tab10 and Set1 for better distinction
    standard_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
    ]
    
    ax = sns.stripplot(
        data=plot_df,
        x="classifier",
        y="rank",
        hue="benchmark_type",
        dodge=True,
        jitter=0.3,
        size=13,
        alpha=0.75,
        linewidth=1.5,
        edgecolor="white",
        palette=standard_colors,
        ax=ax
    )
    
    # Title and labels
    plt.title(
        f"Benchmark Sample Ranks by Classifier\n{num_docs:,} Documents from {dataset_name} • {mode_suffix.capitalize()} Mode", 
        fontsize=18, 
        fontweight='bold', 
        pad=25,
        color='#2c3e50'
    )
    plt.xlabel("Classifier", fontsize=16, fontweight='bold', color='#34495e', labelpad=12)
    plt.ylabel("Rank (0 = best)", fontsize=15, fontweight='semibold', color='#34495e', labelpad=10)
    
    # Make x-axis labels bigger and more readable
    plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    
    # Invert y-axis so 0 is at the top (best rank)
    ax.invert_yaxis()
    
    # Enhanced legend
    plt.legend(
        title="Benchmark Type", 
        title_fontsize=13,
        bbox_to_anchor=(1.01, 1), 
        loc='upper left', 
        frameon=True, 
        shadow=True, 
        fontsize=12,
        fancybox=True,
        edgecolor='#bdc3c7'
    )
    
    # Grid styling
    plt.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8, color='#95a5a6')
    
    # Add vertical lines between classifiers for better separation
    for i in range(len(plot_df['classifier'].unique()) - 1):
        plt.axvline(x=i + 0.5, color='#bdc3c7', linestyle='-', linewidth=1.2, alpha=0.5)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#bdc3c7')
        spine.set_linewidth(1.5)
    
    # Adjust layout to accommodate larger labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    plot_path = os.path.join(results_dir, f"benchmark_ranks_by_classifier{file_suffix}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    console.log(f"[bold green]Saved plot to {plot_path}[/bold green]")

    # Create figure with white background for percentiles
    fig, ax = plt.subplots(figsize=(fig_width, 11), facecolor='white')
    ax.set_facecolor('#f8f9fa')
    
    # Use the same standard colors for consistency
    ax = sns.stripplot(
        data=plot_df,
        x="classifier",
        y="percentile",
        hue="benchmark_type",
        dodge=True,
        jitter=0.3,
        size=13,
        alpha=0.75,
        linewidth=1.5,
        edgecolor="white",
        palette=standard_colors,
        ax=ax
    )
    
    # Title and labels
    plt.title(
        f"Benchmark Sample Percentiles by Classifier\n{num_docs:,} Documents from {dataset_name} • {mode_suffix.capitalize()} Mode", 
        fontsize=18, 
        fontweight='bold', 
        pad=25,
        color='#2c3e50'
    )
    plt.xlabel("Classifier", fontsize=16, fontweight='bold', color='#34495e', labelpad=12)
    plt.ylabel("Percentile (higher is better)", fontsize=15, fontweight='semibold', color='#34495e', labelpad=10)
    
    # Make x-axis labels bigger and more readable
    plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    
    # Enhanced legend
    plt.legend(
        title="Benchmark Type", 
        title_fontsize=13,
        bbox_to_anchor=(1.01, 1), 
        loc='upper left', 
        frameon=True, 
        shadow=True, 
        fontsize=12,
        fancybox=True,
        edgecolor='#bdc3c7'
    )
    
    # Grid styling
    plt.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8, color='#95a5a6')
    
    # Add vertical lines between classifiers for better separation
    for i in range(len(plot_df['classifier'].unique()) - 1):
        plt.axvline(x=i + 0.5, color='#bdc3c7', linestyle='-', linewidth=1.2, alpha=0.5)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#bdc3c7')
        spine.set_linewidth(1.5)
    
    # Adjust layout to accommodate larger labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    plot_path_pct = os.path.join(results_dir, f"benchmark_percentiles_by_classifier{file_suffix}.png")
    plt.savefig(plot_path_pct, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    console.log(f"[bold green]Saved plot to {plot_path_pct}[/bold green]")

def load_cache_data(cache_dir: str, dataset_name: str = None):
    """Load cached classifier results from JSON files.
    
    Args:
        cache_dir: Base cache directory (e.g., 'cache')
        dataset_name: Name of dataset subfolder (e.g., 'fineweb'). If None, auto-detect.
    
    Returns:
        results: Dictionary mapping classifier names to list of score dictionaries
        num_docs: Total number of documents
        inject_inside: Whether benchmarks were injected (inferred from data)
    """
    cache_path = Path(cache_dir)
    
    # Auto-detect dataset subfolder if not specified
    if dataset_name is None:
        subdirs = [d for d in cache_path.iterdir() if d.is_dir() and d.name != 'old']
        if not subdirs:
            raise ValueError(f"No dataset subdirectories found in {cache_dir}")
        if len(subdirs) > 1:
            console.log(f"[yellow]Multiple datasets found: {[d.name for d in subdirs]}[/yellow]")
            console.log(f"[yellow]Using: {subdirs[0].name}[/yellow]")
        dataset_path = subdirs[0]
        dataset_name = dataset_path.name
    else:
        dataset_path = cache_path / dataset_name
        if not dataset_path.exists():
            raise ValueError(f"Dataset directory not found: {dataset_path}")
    
    console.log(f"[cyan]Loading cache from: {dataset_path}[/cyan]")
    
    # Find all classifier JSON files
    json_files = list(dataset_path.glob("*Classifier.json"))
    if not json_files:
        raise ValueError(f"No classifier JSON files found in {dataset_path}")
    
    console.log(f"[green]Found {len(json_files)} classifier cache files[/green]")
    
    results = {}
    num_docs = 0
    
    for json_file in sorted(json_files):
        classifier_name = json_file.stem  # e.g., "DCLMClassifier"
        console.log(f"[yellow]Loading {classifier_name}...[/yellow]")
        
        with open(json_file, 'r') as f:
            cache_data = json.load(f)
        
        # Convert cache format to results format
        scores_list = []
        for doc_hash, doc_data in cache_data.items():
            scores_list.append({
                'doc_hash': doc_hash,
                'id': doc_data['id'],
                'source': doc_data['source'],
                'contains_benchmark': doc_data['contains_benchmark'],
                'benchmark_type': doc_data.get('benchmark_type'),
                'benchmark_index': doc_data.get('benchmark_index'),
                'score': doc_data['score']
            })
        
        results[classifier_name] = scores_list
        num_docs = max(num_docs, len(scores_list))
        console.log(f"[green]  → Loaded {len(scores_list)} documents[/green]")
    
    # Infer inject_inside from data (check if any fineweb docs contain benchmarks)
    inject_inside = False
    for scores in results.values():
        for doc in scores:
            if doc['source'] == 'fineweb' and doc['contains_benchmark']:
                inject_inside = True
                break
        if inject_inside:
            break
    
    console.log(f"[cyan]Total documents: {num_docs}[/cyan]")
    console.log(f"[cyan]Mode: {'injected' if inject_inside else 'separate'}[/cyan]")
    console.log(f"[cyan]Dataset: {dataset_name}[/cyan]")
    
    return results, num_docs, inject_inside, dataset_name

def main():
    """Run analysis standalone from cached data."""
    parser = argparse.ArgumentParser(
        description="Generate analysis plots from cached classifier results"
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='cache',
        help='Base cache directory (default: cache)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset subfolder name (e.g., fineweb). Auto-detect if not specified.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for plots (default: results)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file for additional settings (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    console.rule("[bold blue]Standalone Analysis Mode[/bold blue]")
    
    # Load cached data
    try:
        results, num_docs, inject_inside, dataset_name = load_cache_data(args.cache_dir, args.dataset)
    except Exception as e:
        console.log(f"[bold red]Error loading cache: {e}[/bold red]")
        return 1
    
    # Try to load config for prefilter_hq setting
    prefilter_hq = False
    if os.path.exists(args.config):
        try:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            prefilter_hq = config.get('dataset', {}).get('prefilter_hq', False)
        except Exception as e:
            console.log(f"[yellow]Could not load config: {e}. Using defaults.[/yellow]")
    
    # Generate plots (benchmark_positions not needed for plotting)
    analyze_and_plot(
        results=results,
        documents=None,  # Not needed for plotting from cache
        benchmark_positions={},  # Not needed for plotting from cache
        output_base_dir=args.output_dir,
        inject_inside=inject_inside,
        prefilter_hq=prefilter_hq,
        num_docs=num_docs,
        dataset_name=dataset_name
    )
    
    console.rule("[bold green]Analysis completed successfully![/bold green]")
    return 0

if __name__ == "__main__":
    exit(main())
