"""Benchmark in a Haystack - Visualization"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

CACHE_BASE_DIR = Path("cache")
COLOR_PALETTE = px.colors.qualitative.Set2

def get_available_datasets() -> list[str]:
    """Get list of available datasets from cache subdirectories."""
    if not CACHE_BASE_DIR.exists():
        return []
    return [d.name for d in CACHE_BASE_DIR.iterdir() if d.is_dir()]

def load_cache_files(dataset_name: str = None) -> dict[str, pd.DataFrame]:
    """Load cache files for a specific dataset."""
    if dataset_name:
        cache_dir = CACHE_BASE_DIR / dataset_name
    else:
        # Fallback to old behavior for backwards compatibility
        cache_dir = CACHE_BASE_DIR
    
    if not cache_dir.exists():
        return {}
    
    cache_files = list(cache_dir.glob("*Classifier.json"))
    if not cache_files:
        return {}
    
    classifiers_data = {}
    for cache_file in cache_files:
        classifier_name = cache_file.stem
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            records = [{'doc_hash': doc_hash, 'classifier': classifier_name, **doc_data} 
                      for doc_hash, doc_data in data.items()]
            classifiers_data[classifier_name] = pd.DataFrame(records)
        except Exception as e:
            print(f"Error loading {cache_file}: {e}")
    return classifiers_data

def load_data(dataset_name: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data for a specific dataset."""
    classifiers_data = load_cache_files(dataset_name)
    if not classifiers_data:
        return pd.DataFrame(), pd.DataFrame()
    
    combined = pd.concat(classifiers_data.values(), ignore_index=True)
    combined['score'] = pd.to_numeric(combined['score'], errors='coerce')
    combined['rank'] = combined.groupby('classifier')['score'].rank(ascending=False, method='min')
    combined['percentile'] = combined.groupby('classifier')['rank'].transform(
        lambda x: (x.max() - x + 1) / x.max() * 100
    )
    
    benchmark_df = combined[combined['contains_benchmark'] == True].copy()
    return combined, benchmark_df

def plot_comparison(benchmark_df: pd.DataFrame, 
                   selected_benchmarks: list[str],
                   selected_classifiers: list[str],
                   metric: str) -> go.Figure:
    if benchmark_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        return fig
    
    df = benchmark_df.copy()
    if selected_benchmarks and "All" not in selected_benchmarks:
        df = df[df['benchmark_type'].isin(selected_benchmarks)]
    if selected_classifiers and "All" not in selected_classifiers:
        df = df[df['classifier'].isin(selected_classifiers)]
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data matching filters", showarrow=False, font=dict(size=16))
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Get unique benchmark types and assign colors
    benchmark_types = df['benchmark_type'].unique()
    colors = {bt: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, bt in enumerate(benchmark_types)}
    
    # Add scatter points for each benchmark type
    for benchmark_type in benchmark_types:
        df_subset = df[df['benchmark_type'] == benchmark_type]
        
        fig.add_trace(go.Scatter(
            x=df_subset['classifier'],
            y=df_subset[metric],
            mode='markers',
            name=benchmark_type,
            marker=dict(size=10, color=colors[benchmark_type], opacity=0.7),
            hovertemplate='<b>%{x}</b><br>' +
                         f'{metric.capitalize()}: %{{y}}<br>' +
                         '<extra></extra>'
        ))
    
    # Update layout
    y_label = "Rank (lower is better)" if metric == "rank" else "Percentile (higher is better)"
    
    fig.update_layout(
        title=f"Benchmark Rankings Across Classifiers ({metric.capitalize()})",
        xaxis_title="Classifier",
        yaxis_title=y_label,
        xaxis=dict(tickangle=45),
        width=1400,
        height=800,
        hovermode='closest',
        showlegend=True,
        template='plotly_white'
    )
    
    # Reverse Y-axis for rank (lower is better)
    if metric == "rank":
        fig.update_yaxes(autorange="reversed")
    
    return fig

def create_summary_table(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    if benchmark_df.empty:
        return pd.DataFrame()
    
    stats = benchmark_df.groupby('classifier').agg({
        'rank': ['mean', 'median', 'min', 'max'],
        'percentile': ['mean', 'median'],
        'score': ['mean', 'median']
    }).round(2)
    
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    stats.columns = [
        'Classifier', 'Mean Rank', 'Median Rank', 'Best Rank', 'Worst Rank',
        'Mean Percentile', 'Median Percentile', 'Mean Score', 'Median Score'
    ]
    return stats.sort_values('Mean Rank')

def create_app():
    print("Loading available datasets...")
    available_datasets = get_available_datasets()
    
    if not available_datasets:
        print(f"⚠️  No datasets found in {CACHE_BASE_DIR.absolute()}")
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown(f"# ⚠️ No Data Found\n\nNo dataset cache folders in `{CACHE_BASE_DIR.absolute()}`\n\n"
                       f"Run the haystack experiment first to generate cache data.")
        return app
    
    print(f"Found datasets: {', '.join(available_datasets)}")
    default_dataset = available_datasets[0]
    
    print(f"Loading data for dataset: {default_dataset}...")
    combined_df, benchmark_df = load_data(default_dataset)
    
    if combined_df.empty:
        print(f"⚠️  No data found for dataset {default_dataset}")
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown(f"# ⚠️ No Data Found\n\nNo cache files in `{CACHE_BASE_DIR.absolute()}/{default_dataset}`")
        return app
    
    classifiers = sorted(combined_df['classifier'].unique().tolist())
    benchmark_types = sorted(benchmark_df['benchmark_type'].unique().tolist())
    
    print("✓ Data loaded successfully\n")
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Benchmark in a Haystack") as app:
        gr.Markdown("# 📊 Benchmark in a Haystack\n\nCompare how quality classifiers rank benchmark samples.")
        
        with gr.Row():
            with gr.Column(scale=1):
                dataset_dropdown = gr.Dropdown(
                    choices=available_datasets,
                    value=default_dataset,
                    label="Dataset",
                    info="Select which dataset cache to visualize"
                )
                metric_radio = gr.Radio(
                    choices=["rank", "percentile"],
                    value="rank",
                    label="Metric"
                )
                benchmark_filter = gr.CheckboxGroup(
                    choices=["All"] + benchmark_types,
                    value=["All"],
                    label="Benchmark Types"
                )
                classifier_filter = gr.CheckboxGroup(
                    choices=["All"] + classifiers,
                    value=["All"],
                    label="Classifiers"
                )
                refresh_btn = gr.Button("🔄 Refresh", variant="primary")
            
            with gr.Column(scale=3):
                comparison_plot = gr.Plot(
                    label="Classifier Comparison",
                    show_label=True
                )
        
        gr.Markdown("### Summary Statistics")
        summary_table = gr.Dataframe(
            value=create_summary_table(benchmark_df),
            label="Performance by Classifier",
            interactive=False
        )
        
        # State to store current dataset data
        current_data = gr.State((combined_df, benchmark_df, classifiers, benchmark_types))
        
        def update_dataset(dataset_name):
            """Load new dataset and update all components."""
            combined, benchmark = load_data(dataset_name)
            if combined.empty:
                return (
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    go.Figure().add_annotation(text=f"No data for {dataset_name}", showarrow=False),
                    pd.DataFrame(),
                    (combined, benchmark, [], [])
                )
            
            clfs = sorted(combined['classifier'].unique().tolist())
            bench_types = sorted(benchmark['benchmark_type'].unique().tolist())
            
            return (
                gr.update(choices=["All"] + bench_types, value=["All"]),
                gr.update(choices=["All"] + clfs, value=["All"]),
                plot_comparison(benchmark, ["All"], ["All"], "rank"),
                create_summary_table(benchmark),
                (combined, benchmark, clfs, bench_types)
            )
        
        def update_plot(metric, bench_filter, clf_filter, data_state):
            """Update plot based on filters."""
            _, benchmark, _, _ = data_state
            return plot_comparison(benchmark, bench_filter, clf_filter, metric)
        
        dataset_dropdown.change(
            fn=update_dataset,
            inputs=[dataset_dropdown],
            outputs=[benchmark_filter, classifier_filter, comparison_plot, summary_table, current_data]
        )
        
        refresh_btn.click(
            fn=update_plot,
            inputs=[metric_radio, benchmark_filter, classifier_filter, current_data],
            outputs=[comparison_plot]
        )
        
        comparison_plot.value = plot_comparison(benchmark_df, ["All"], ["All"], "rank")
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
