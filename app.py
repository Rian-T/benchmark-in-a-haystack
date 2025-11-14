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
# Use the same standard colors as analysis.py matplotlib plots
COLOR_PALETTE = [
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

def get_available_datasets() -> list[str]:
    """Get list of available datasets from cache subdirectories."""
    if not CACHE_BASE_DIR.exists():
        return []
    return [d.name for d in CACHE_BASE_DIR.iterdir() if d.is_dir()]

def load_cached_document_texts(dataset_name: str) -> dict[str, str]:
    """Load cached document texts from the top_documents_texts.json file.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'fineweb')
    
    Returns:
        Dictionary mapping doc_id to text content
    """
    cache_file = CACHE_BASE_DIR / dataset_name / "top_documents_texts.json"
    
    if not cache_file.exists():
        print(f"⚠️  No cached texts found at {cache_file}")
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading cached texts: {e}")
        return {}

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
    
    # Match analysis.py styling
    if metric == "rank":
        y_label = "Rank (0 = best)"
        title_text = "Benchmark Sample Ranks by Classifier"
    else:
        y_label = "Percentile (higher is better)"
        title_text = "Benchmark Sample Percentiles by Classifier"
    
    fig = px.strip(
        df, 
        x='classifier', 
        y=metric, 
        color='benchmark_type',
        hover_data=['id', 'score', 'rank', 'percentile'],
        color_discrete_sequence=COLOR_PALETTE,
    )
    
    # Update markers to match matplotlib style
    fig.update_traces(
        marker=dict(
            size=13,  # Larger markers like matplotlib
            opacity=0.75,
            line=dict(width=1.5, color='white')  # White edge like matplotlib
        ),
        jitter=0.3  # Add jitter like matplotlib stripplot
    )
    
    # Get number of documents for subtitle
    num_docs = len(df.groupby(['classifier', 'id']).first())
    
    # Update layout to match matplotlib style from analysis.py
    fig.update_layout(
        title={
            'text': title_text,
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.98,
            'yanchor': 'top'
        },
        xaxis_title={
            'text': "Classifier",
            'font': {'size': 16, 'color': '#34495e', 'family': 'Arial, sans-serif'}
        },
        yaxis_title={
            'text': y_label,
            'font': {'size': 15, 'color': '#34495e', 'family': 'Arial, sans-serif'}
        },
        hovermode='closest',
        width=1400,
        height=750,
        plot_bgcolor='#f8f9fa',  # Light gray background like matplotlib
        paper_bgcolor='white',
        font={'family': 'Arial, sans-serif', 'size': 12},
        xaxis=dict(
            tickangle=45,
            tickfont={'size': 14, 'color': '#2c3e50'},
            showgrid=False,
            showline=True,
            linewidth=1.5,
            linecolor='#bdc3c7',
            mirror=True
        ),
        yaxis=dict(
            tickfont={'size': 12, 'color': '#2c3e50'},
            showgrid=True,
            gridcolor='#95a5a6',
            gridwidth=0.8,
            griddash='dash',
            showline=True,
            linewidth=1.5,
            linecolor='#bdc3c7',
            mirror=True
        ),
        legend=dict(
            title={'text': "Benchmark Type", 'font': {'size': 13, 'color': '#2c3e50'}},
            orientation="v",
            x=1.01,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='white',
            bordercolor='#bdc3c7',
            borderwidth=1.5,
            font={'size': 12}
        ),
        margin=dict(t=80, b=100, l=80, r=150)
    )
    
    # Reverse Y-axis for rank (0 at top, like matplotlib)
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

def get_top_documents_per_classifier(combined_df: pd.DataFrame, dataset_name: str, top_n: int = 10) -> dict[str, str]:
    """Get the top N highest-scoring documents for each classifier.
    
    Args:
        combined_df: DataFrame with all scored documents
        dataset_name: Name of the dataset to load texts from
        top_n: Number of top documents to retrieve per classifier
    
    Returns:
        Dictionary mapping classifier name to plain text string with formatted documents
    """
    if combined_df.empty:
        return {}
    
    # Get unique classifiers
    classifiers = sorted(combined_df['classifier'].unique())
    
    # Collect all doc IDs we need to load
    all_doc_ids = set()
    top_docs_by_classifier = {}
    
    for classifier in classifiers:
        clf_data = combined_df[combined_df['classifier'] == classifier].copy()
        clf_data = clf_data.nlargest(top_n, 'score')
        top_docs_by_classifier[classifier] = clf_data
        all_doc_ids.update(clf_data['id'].tolist())
    
    # Load texts from cache
    doc_texts = load_cached_document_texts(dataset_name)
    
    # Build simple text output for each classifier
    result = {}
    
    for classifier in classifiers:
        clf_data = top_docs_by_classifier[classifier]
        
        # Calculate min and max scores for this classifier
        clf_all_data = combined_df[combined_df['classifier'] == classifier]
        min_score = clf_all_data['score'].min()
        max_score = clf_all_data['score'].max()
        
        text_parts = []
        text_parts.append(f"Score Range: {min_score:.6f} (min) to {max_score:.6f} (max)\n")
        
        for top_rank, (idx, row) in enumerate(clf_data.iterrows(), start=1):
            doc_id = row['id']
            score = row['score']
            is_benchmark = row.get('contains_benchmark', False)
            benchmark_type = row.get('benchmark_type', 'N/A')
            
            # Get full text (no truncation)
            text = doc_texts.get(doc_id, "[Text not cached - run haystack.py to cache top documents]")
            
            # Create badge
            badge = "🔴 BENCHMARK" if is_benchmark else "🟢 Regular"
            benchmark_info = f" | Type: {benchmark_type}" if is_benchmark else ""
            
            # Document header
            text_parts.append(f"\n{'-'*100}")
            text_parts.append(f"Top {top_rank} | {classifier} | {badge} | ID: {doc_id} | Score: {score:.6f} | Range: {min_score:.6f}–{max_score:.6f}{benchmark_info}")
            text_parts.append(f"{'-'*100}")
            text_parts.append(text)
            text_parts.append("")
        
        result[classifier] = "\n".join(text_parts)
    
    return result

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
    
    # Preload ALL datasets
    print("Preloading all datasets for instant switching...")
    all_datasets_data = {}
    for dataset_name in available_datasets:
        print(f"  Loading {dataset_name}...")
        combined_df, benchmark_df = load_data(dataset_name)
        if not combined_df.empty:
            classifiers = sorted(combined_df['classifier'].unique().tolist())
            benchmark_types = sorted(benchmark_df['benchmark_type'].unique().tolist())
            all_datasets_data[dataset_name] = {
                'combined': combined_df,
                'benchmark': benchmark_df,
                'classifiers': classifiers,
                'benchmark_types': benchmark_types
            }
        else:
            print(f"    ⚠️  No data found for {dataset_name}")
    
    if not all_datasets_data:
        print(f"⚠️  No valid data found in any dataset")
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown(f"# ⚠️ No Data Found\n\nNo cache files found in any dataset folder")
        return app
    
    print("✓ All datasets loaded successfully\n")
    
    # Use first dataset with data as default
    default_dataset = list(all_datasets_data.keys())[0]
    combined_df = all_datasets_data[default_dataset]['combined']
    benchmark_df = all_datasets_data[default_dataset]['benchmark']
    classifiers = all_datasets_data[default_dataset]['classifiers']
    benchmark_types = all_datasets_data[default_dataset]['benchmark_types']
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Benchmark in a Haystack") as app:
        gr.Image("biahs-banner.png", show_label=False, container=False, show_download_button=False, width=600)
        gr.Markdown("Compare how quality classifiers rank benchmark samples.")
        
        with gr.Row():
            with gr.Column(scale=1):
                dataset_dropdown = gr.Dropdown(
                    choices=list(all_datasets_data.keys()),
                    value=default_dataset,
                    label="Dataset",
                    info="Select the dataset to use as the haystack"
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
                    value=plot_comparison(benchmark_df, ["All"], ["All"], "rank"),
                    label="Classifier Comparison",
                    show_label=True
                )
        
        gr.Markdown("### Summary Statistics")
        summary_table = gr.Dataframe(
            value=create_summary_table(benchmark_df),
            label="Performance by Classifier",
            interactive=False
        )
        
        gr.Markdown("### Top 10 Highest-Scoring Documents per Classifier")
        
        # Get initial top documents data
        initial_docs = get_top_documents_per_classifier(combined_df, default_dataset, top_n=10)
        
        # Create separate textbox for each classifier
        classifier_textboxes = {}
        for classifier in classifiers:
            gr.Markdown(f"#### {classifier}")
            classifier_textboxes[classifier] = gr.Textbox(
                value=initial_docs.get(classifier, "No data"),
                lines=30,
                max_lines=50,
                show_label=False,
                interactive=False
            )
        
        # State to store all preloaded datasets and current dataset info
        all_data_state = gr.State(all_datasets_data)
        current_data = gr.State((combined_df, benchmark_df, classifiers, benchmark_types, default_dataset))
        
        def update_dataset(dataset_name, all_datasets):
            """Switch to a different preloaded dataset (instant)."""
            if dataset_name not in all_datasets:
                empty_results = [
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    go.Figure().add_annotation(text=f"No data for {dataset_name}", showarrow=False),
                    pd.DataFrame(),
                    (pd.DataFrame(), pd.DataFrame(), [], [], dataset_name)
                ]
                # Add empty update for each classifier textbox
                for _ in classifiers:
                    empty_results.append("No data available")
                return tuple(empty_results)
            
            # Get preloaded data (instant!)
            data = all_datasets[dataset_name]
            combined = data['combined']
            benchmark = data['benchmark']
            clfs = data['classifiers']
            bench_types = data['benchmark_types']
            
            # Get documents for each classifier
            docs_by_classifier = get_top_documents_per_classifier(combined, dataset_name, top_n=10)
            
            results = [
                gr.update(choices=["All"] + bench_types, value=["All"]),
                gr.update(choices=["All"] + clfs, value=["All"]),
                plot_comparison(benchmark, ["All"], ["All"], "rank"),
                create_summary_table(benchmark),
                (combined, benchmark, clfs, bench_types, dataset_name)
            ]
            
            # Add textbox update for each classifier
            for clf in classifiers:
                results.append(docs_by_classifier.get(clf, "No data"))
            
            return tuple(results)
        
        def update_plot(metric, bench_filter, clf_filter, data_state):
            """Update plot based on filters."""
            _, benchmark, _, _, _ = data_state
            return plot_comparison(benchmark, bench_filter, clf_filter, metric)
        
        # Build outputs list with all classifier textboxes
        outputs_list = [benchmark_filter, classifier_filter, comparison_plot, summary_table, current_data]
        outputs_list.extend(list(classifier_textboxes.values()))
        
        dataset_dropdown.change(
            fn=update_dataset,
            inputs=[dataset_dropdown, all_data_state],
            outputs=outputs_list
        )
        
        # Auto-update plot when filters or metric changes
        metric_radio.change(
            fn=update_plot,
            inputs=[metric_radio, benchmark_filter, classifier_filter, current_data],
            outputs=[comparison_plot]
        )
        
        benchmark_filter.change(
            fn=update_plot,
            inputs=[metric_radio, benchmark_filter, classifier_filter, current_data],
            outputs=[comparison_plot]
        )
        
        classifier_filter.change(
            fn=update_plot,
            inputs=[metric_radio, benchmark_filter, classifier_filter, current_data],
            outputs=[comparison_plot]
        )
        
        refresh_btn.click(
            fn=update_plot,
            inputs=[metric_radio, benchmark_filter, classifier_filter, current_data],
            outputs=[comparison_plot]
        )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
