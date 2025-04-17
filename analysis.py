import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from rich.console import Console

console = Console()

def analyze_and_plot(results, documents, benchmark_positions, inject_inside=True, prefilter_hq=False):
    """
    For each classifier, output a single JSON of benchmark sample ranks across classifiers.
    Also plot benchmark sample ranks.
    """
    # --- Create timestamped results directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

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

        # Find benchmark samples and their ranks
        bench_df = scores_df[scores_df["contains_benchmark"] == True].copy()
        bench_df["classifier"] = clf_name
        bench_df["percentile"] = (len(scores_df) - bench_df["rank"]) / len(scores_df) * 100

        # For single JSON: collect by (id, benchmark_type, benchmark_index)
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

    # Output single JSON file
    bench_ranks_json = os.path.join(results_dir, "benchmark_ranks_all_classifiers.json")
    with open(bench_ranks_json, "w") as f:
        json.dump(list(bench_ranks_dict.values()), f, indent=2)
    console.log(f"[green]Saved all benchmark ranks to {bench_ranks_json}[/green]")

    # --- Improved Plot: Benchmark Ranks by Classifier and Type ---
    # Prepare data for plotting: one row per benchmark sample per classifier
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

    # Plot: Horizontal strip plot of benchmark ranks per classifier, grouped by benchmark type
    console.log("[yellow]Plotting benchmark sample ranks by classifier and benchmark type...[/yellow]")
    plt.figure(figsize=(10, 6))
    ax = sns.stripplot(
        data=plot_df,
        x="rank",
        y="classifier",
        hue="benchmark_type",
        dodge=True,
        jitter=True,
        size=8,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="gray"
    )
    plt.title("Benchmark Sample Ranks by Classifier")
    plt.xlabel("Rank (lower is better)")
    plt.ylabel("Classifier")
    plt.legend(title="Benchmark Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "benchmark_ranks_by_classifier.png")
    plt.savefig(plot_path)
    plt.close()
    console.log(f"[bold green]Saved plot to {plot_path}[/bold green]")

    # Optional: also plot percentiles for a normalized view
    plt.figure(figsize=(10, 6))
    ax = sns.stripplot(
        data=plot_df,
        x="percentile",
        y="classifier",
        hue="benchmark_type",
        dodge=True,
        jitter=True,
        size=8,
        alpha=0.8,
        linewidth=0.5,
        edgecolor="gray"
    )
    plt.title("Benchmark Sample Percentiles by Classifier")
    plt.xlabel("Percentile (higher is better)")
    plt.ylabel("Classifier")
    plt.legend(title="Benchmark Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_path_pct = os.path.join(results_dir, "benchmark_percentiles_by_classifier.png")
    plt.savefig(plot_path_pct)
    plt.close()
    console.log(f"[bold green]Saved plot to {plot_path_pct}[/bold green]")
