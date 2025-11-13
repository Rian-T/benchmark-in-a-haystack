import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from rich.console import Console

console = Console()

def analyze_and_plot(results, documents, benchmark_positions, output_base_dir="results", inject_inside=True, prefilter_hq=False, num_docs=100000):
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
    fig_height = max(8, num_classifiers * 0.8)
    
    plt.figure(figsize=(14, fig_height))
    ax = sns.stripplot(
        data=plot_df,
        x="rank",
        y="classifier",
        hue="benchmark_type",
        dodge=True,
        jitter=0.3,
        size=10,
        alpha=0.75,
        linewidth=1,
        edgecolor="black"
    )
    plt.title(f"Benchmark Sample Ranks by Classifier ({mode_suffix.capitalize()}, {num_docs:,} docs)", fontsize=14, fontweight='bold')
    plt.xlabel("Rank (lower is better)", fontsize=12)
    plt.ylabel("Classifier", fontsize=12)
    plt.legend(title="Benchmark Type", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"benchmark_ranks_by_classifier{file_suffix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    console.log(f"[bold green]Saved plot to {plot_path}[/bold green]")

    plt.figure(figsize=(14, fig_height))
    ax = sns.stripplot(
        data=plot_df,
        x="percentile",
        y="classifier",
        hue="benchmark_type",
        dodge=True,
        jitter=0.3,
        size=10,
        alpha=0.75,
        linewidth=1,
        edgecolor="black"
    )
    plt.title(f"Benchmark Sample Percentiles by Classifier ({mode_suffix.capitalize()}, {num_docs:,} docs)", fontsize=14, fontweight='bold')
    plt.xlabel("Percentile (higher is better)", fontsize=12)
    plt.ylabel("Classifier", fontsize=12)
    plt.legend(title="Benchmark Type", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot_path_pct = os.path.join(results_dir, f"benchmark_percentiles_by_classifier{file_suffix}.png")
    plt.savefig(plot_path_pct, dpi=150, bbox_inches='tight')
    plt.close()
    console.log(f"[bold green]Saved plot to {plot_path_pct}[/bold green]")
