"""Benchmark in a Haystack - Visualization"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
from models import DCLMClassifier

CACHE_BASE_DIR = Path("cache")
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
BENCHMARK_ORDER = ['gpqa', 'mmlu', 'gsm8k']
BENCHMARK_COLORS = {
    'gpqa': '#1f77b4',
    'mmlu': '#ff7f0e',
    'gsm8k': '#2ca02c',
    'inference': '#e74c3c',
}

def get_available_datasets() -> list[str]:
    """Get list of available datasets from cache subdirectories."""
    if not CACHE_BASE_DIR.exists():
        return []
    return [d.name for d in CACHE_BASE_DIR.iterdir() if d.is_dir()]

def load_cached_document_texts(dataset_name: str) -> dict[str, str]:
    """Load cached document texts from the top_documents_texts.json file."""
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
    cache_dir = CACHE_BASE_DIR / dataset_name if dataset_name else CACHE_BASE_DIR
    
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
                   metric: str,
                   dataset_name: str = "") -> go.Figure:
    if benchmark_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False, font=dict(size=16))
        return fig
    
    df = benchmark_df.copy()
    if selected_benchmarks and "All" not in selected_benchmarks:
        if "Gaperon paper" in selected_benchmarks:
            gaperon_benchmarks = ['mmlu', 'gsm8k', 'gpqa']
            other_benchmarks = [b for b in selected_benchmarks if b != "Gaperon paper"]
            combined_benchmarks = gaperon_benchmarks + other_benchmarks
            df = df[df['benchmark_type'].isin(combined_benchmarks)]
        else:
            df = df[df['benchmark_type'].isin(selected_benchmarks)]
    if selected_classifiers and "All" not in selected_classifiers:
        if "Gaperon paper" in selected_classifiers:
            gaperon_classifiers = ['GaperonClassifier', 'FinewebEduClassifier', 'DCLMClassifier', 'TextbookFastTextClassifier']
            other_classifiers = [c for c in selected_classifiers if c != "Gaperon paper"]
            combined_classifiers = gaperon_classifiers + other_classifiers
            df = df[df['classifier'].isin(combined_classifiers)]
        else:
            df = df[df['classifier'].isin(selected_classifiers)]
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data matching filters", showarrow=False, font=dict(size=16))
        return fig
    
    if metric == "rank":
        x_label = "Rank (0 = best)"
        title_text = "Benchmark Sample Ranks by Classifier"
    else:
        x_label = "Percentile (higher is better)"
        title_text = "Benchmark Sample Percentiles by Classifier"
    
    subtitle_text = f"Haystack: {dataset_name} (100k documents)" if dataset_name else ""
    
    gaperon_order = ['GaperonClassifier', 'FinewebEduClassifier', 'DCLMClassifier', 'TextbookFastTextClassifier']
    all_classifiers = df['classifier'].unique().tolist()
    classifier_order = [c for c in gaperon_order if c in all_classifiers]
    other_clfs = [c for c in all_classifiers if c not in gaperon_order]
    classifier_order.extend(other_clfs)
    
    all_benchmarks = df['benchmark_type'].unique().tolist()
    benchmark_order = [b for b in BENCHMARK_ORDER if b in all_benchmarks]
    other_benchmarks = [b for b in all_benchmarks if b not in BENCHMARK_ORDER]
    benchmark_order.extend(other_benchmarks)
    
    color_map = BENCHMARK_COLORS.copy()
    extra_colors = [c for c in COLOR_PALETTE if c not in BENCHMARK_COLORS.values()]
    for i, bench in enumerate(other_benchmarks):
        if bench not in color_map:
            color_map[bench] = extra_colors[i % len(extra_colors)]
    
    has_inference = 'inference' in df['benchmark_type'].values
    if has_inference:
        df_regular = df[df['benchmark_type'] != 'inference'].copy()
        df_inference = df[df['benchmark_type'] == 'inference'].copy()
    else:
        df_regular = df.copy()
        df_inference = pd.DataFrame()
    
    fig = px.strip(
        df_regular, 
        y='classifier',
        x=metric,
        color='benchmark_type',
        hover_data=['id', 'score', 'rank', 'percentile'],
        color_discrete_map=color_map,
        category_orders={'classifier': classifier_order, 'benchmark_type': benchmark_order}
    )
    
    fig.update_traces(
        marker=dict(size=13, opacity=0.75, line=dict(width=1.5, color='white')),
        jitter=0.3
    )
    
    if has_inference and not df_inference.empty:
        for _, row in df_inference.iterrows():
            fig.add_trace(go.Box(
                x=[row[metric]],
                y=[row['classifier']],
                name='user text',
                marker=dict(
                    color='#e74c3c',
                    size=13,
                    symbol='star',
                    line=dict(color='black', width=1.5)
                ),
                boxpoints='all',
                jitter=0,
                pointpos=0,
                fillcolor='rgba(0,0,0,0)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=True,
                hovertemplate=f'user text<br>Classifier: {row["classifier"]}<br>Score: {row["score"]:.6f}<br>Rank: {row["rank"]:.0f}<br>Percentile: {row["percentile"]:.1f}<extra></extra>'
            ))
    
    
    fig.update_layout(
        title={
            'text': f"{title_text}<br><span style='font-size:14px'>{subtitle_text}</span>" if subtitle_text else title_text,
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial, sans-serif'},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top',
            'pad': {'b': 10}
        },
        yaxis_title={
            'text': "Classifier",
            'font': {'size': 16, 'color': '#34495e', 'family': 'Arial, sans-serif'}
        },
        xaxis_title={
            'text': x_label,
            'font': {'size': 15, 'color': '#34495e', 'family': 'Arial, sans-serif'}
        },
        hovermode='closest',
        height=750,
        autosize=True,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        font={'family': 'Arial, sans-serif', 'size': 12},
        yaxis=dict(
            tickfont={'size': 14, 'color': '#2c3e50'},
            showgrid=False,
            showline=True,
            linewidth=1.5,
            linecolor='#bdc3c7',
            mirror=True
        ),
        xaxis=dict(
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
            x=0.99,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='white',
            bordercolor='#bdc3c7',
            borderwidth=1.5,
            font={'size': 12},
            traceorder='normal'
        ),
        margin=dict(t=110, b=100, l=150, r=150)
    )
    
    num_classifiers = len(df['classifier'].unique())
    for i in range(num_classifiers - 1):
        fig.add_hline(
            y=i + 0.5,
            line_color='#bdc3c7',
            line_width=1.2,
            opacity=0.5
        )
    
    trace_order = {bench: i for i, bench in enumerate(benchmark_order)}
    fig.data = sorted(fig.data, key=lambda trace: trace_order.get(trace.name, 999))
    
    if metric == "rank":
        fig.update_xaxes(autorange="reversed")
    
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
    """Get the top N highest-scoring documents for each classifier."""
    if combined_df.empty:
        return {}
    
    classifiers = sorted(combined_df['classifier'].unique())
    all_doc_ids = set()
    top_docs_by_classifier = {}
    
    for classifier in classifiers:
        clf_data = combined_df[combined_df['classifier'] == classifier].copy()
        clf_data = clf_data.nlargest(top_n, 'score')
        top_docs_by_classifier[classifier] = clf_data
        all_doc_ids.update(clf_data['id'].tolist())
    
    doc_texts = load_cached_document_texts(dataset_name)
    result = {}
    
    for classifier in classifiers:
        clf_data = top_docs_by_classifier[classifier]
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
            
            text = doc_texts.get(doc_id, "[Text not cached - run haystack.py to cache top documents]")
            badge = "🔴 BENCHMARK" if is_benchmark else "🟢 Regular"
            benchmark_info = f" | Type: {benchmark_type}" if is_benchmark else ""
            
            text_parts.append(f"\n{'-'*100}")
            text_parts.append(f"Top {top_rank} | {classifier} | {badge} | ID: {doc_id} | Score: {score:.6f} | Range: {min_score:.6f}–{max_score:.6f}{benchmark_info}")
            text_parts.append(f"{'-'*100}")
            text_parts.append(text)
            text_parts.append("")
        
        result[classifier] = "\n".join(text_parts)
    
    return result

def perform_inference(text_input, benchmark_df, metric, bench_filter, clf_filter, dataset_name, dclm_model):
    """Perform real-time inference on user text with DCLM FastText classifier."""
    if not text_input or not text_input.strip():
        return plot_comparison(benchmark_df, bench_filter, clf_filter, metric, dataset_name)
    
    doc = {
        "id": "inference-result",
        "text": text_input.strip(),
        "source": "user-input",
        "contains_benchmark": False,
        "benchmark_type": "inference",
        "benchmark_index": None
    }
    
    dclm_results = dclm_model._score_documents([doc])
    
    inference_rows = []
    result = dclm_results[0]
    inference_rows.append({
        'doc_hash': 'inference',
        'classifier': 'DCLMClassifier',
        'id': result['id'],
        'source': result['source'],
        'contains_benchmark': result['contains_benchmark'],
        'benchmark_type': result['benchmark_type'],
        'benchmark_index': result['benchmark_index'],
        'score': result['score'],
        'rank': 1,
        'percentile': 100
    })
    
    inference_df = pd.DataFrame(inference_rows)
    combined_df = pd.concat([benchmark_df, inference_df], ignore_index=True)
    combined_df['rank'] = combined_df.groupby('classifier')['score'].rank(ascending=False, method='min')
    combined_df['percentile'] = combined_df.groupby('classifier')['rank'].transform(
        lambda x: (x.max() - x + 1) / x.max() * 100
    )
    
    return plot_comparison(combined_df, bench_filter, clf_filter, metric, dataset_name)

def create_app():
    print("Loading available datasets...")
    available_datasets = get_available_datasets()
    
    print("Initializing inference model (DCLM)...")
    try:
        dclm_classifier = DCLMClassifier()
        print("✓ Inference model loaded successfully\n")
    except Exception as e:
        print(f"⚠️  Error loading inference model: {e}")
        dclm_classifier = None
    
    if not available_datasets:
        print(f"⚠️  No datasets found in {CACHE_BASE_DIR.absolute()}")
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown(f"# ⚠️ No Data Found\n\nNo dataset cache folders in `{CACHE_BASE_DIR.absolute()}`\n\n"
                       f"Run the haystack experiment first to generate cache data.")
        return app
    
    print(f"Found datasets: {', '.join(available_datasets)}")
    
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
    
    default_dataset = list(all_datasets_data.keys())[0]
    combined_df = all_datasets_data[default_dataset]['combined']
    benchmark_df = all_datasets_data[default_dataset]['benchmark']
    classifiers = all_datasets_data[default_dataset]['classifiers']
    benchmark_types = all_datasets_data[default_dataset]['benchmark_types']
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Benchmark in a Haystack") as app:
        gr.Image("biahs-banner.png", show_label=False, show_download_button=False, width=800)
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
                    choices=["All", "Gaperon paper"] + benchmark_types,
                    value=["All"],
                    label="Benchmark Types"
                )
                classifier_filter = gr.CheckboxGroup(
                    choices=["All", "Gaperon paper"] + classifiers,
                    value=["All"],
                    label="Classifiers"
                )
                refresh_btn = gr.Button("🔄 Refresh", variant="primary")
            
            with gr.Column(scale=3):
                comparison_plot = gr.Plot(
                    value=plot_comparison(benchmark_df, ["All"], ["All"], "rank", default_dataset),
                    label="Classifier Comparison",
                    show_label=True
                )
        
        gr.Markdown("### Real-Time Inference")
        gr.Markdown("Enter text below to see how DCLMClassifier scores it in real-time (CPU inference).")
        inference_input = gr.Textbox(
            label="Input Text",
            placeholder="Type or paste text here for real-time inference...",
            lines=10,
            max_lines=20,
            interactive=True
        )
        
        gr.Markdown("### Summary Statistics")
        summary_table = gr.Dataframe(
            value=create_summary_table(benchmark_df),
            label="Performance by Classifier",
            interactive=False
        )
        
        gr.Markdown("### Top 10 Highest-Scoring Documents per Classifier")
        
        initial_docs = get_top_documents_per_classifier(combined_df, default_dataset, top_n=10)
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
                for _ in classifiers:
                    empty_results.append("No data available")
                return tuple(empty_results)
            
            data = all_datasets[dataset_name]
            combined = data['combined']
            benchmark = data['benchmark']
            clfs = data['classifiers']
            bench_types = data['benchmark_types']
            
            docs_by_classifier = get_top_documents_per_classifier(combined, dataset_name, top_n=10)
            
            results = [
                gr.update(choices=["All", "Gaperon paper"] + bench_types, value=["All"]),
                gr.update(choices=["All", "Gaperon paper"] + clfs, value=["All"]),
                plot_comparison(benchmark, ["All"], ["All"], "rank", dataset_name),
                create_summary_table(benchmark),
                (combined, benchmark, clfs, bench_types, dataset_name)
            ]
            
            for clf in classifiers:
                results.append(docs_by_classifier.get(clf, "No data"))
            
            return tuple(results)
        
        def update_plot(metric, bench_filter, clf_filter, data_state):
            """Update plot based on filters."""
            _, benchmark, _, _, dataset_name = data_state
            return plot_comparison(benchmark, bench_filter, clf_filter, metric, dataset_name)
        
        def handle_benchmark_selection(selected):
            """Handle exclusive selection for All/Gaperon paper in benchmarks."""
            if not selected:
                return gr.update(value=["All"])
            if "All" in selected and len(selected) > 1:
                if selected[-1] == "All":
                    return gr.update(value=["All"])
                else:
                    return gr.update(value=[s for s in selected if s != "All"])
            if "Gaperon paper" in selected and len(selected) > 1:
                if selected[-1] == "Gaperon paper":
                    return gr.update(value=["Gaperon paper"])
                else:
                    return gr.update(value=[s for s in selected if s != "Gaperon paper"])
            return gr.update(value=selected)
        
        def handle_classifier_selection(selected):
            """Handle exclusive selection for All/Gaperon paper in classifiers."""
            if not selected:
                return gr.update(value=["All"])
            if "All" in selected and len(selected) > 1:
                if selected[-1] == "All":
                    return gr.update(value=["All"])
                else:
                    return gr.update(value=[s for s in selected if s != "All"])
            if "Gaperon paper" in selected and len(selected) > 1:
                if selected[-1] == "Gaperon paper":
                    return gr.update(value=["Gaperon paper"])
                else:
                    return gr.update(value=[s for s in selected if s != "Gaperon paper"])
            return gr.update(value=selected)
        
        outputs_list = [benchmark_filter, classifier_filter, comparison_plot, summary_table, current_data]
        outputs_list.extend(list(classifier_textboxes.values()))
        
        dataset_dropdown.change(
            fn=update_dataset,
            inputs=[dataset_dropdown, all_data_state],
            outputs=outputs_list
        )
        
        metric_radio.change(
            fn=update_plot,
            inputs=[metric_radio, benchmark_filter, classifier_filter, current_data],
            outputs=[comparison_plot]
        )
        
        benchmark_filter.change(
            fn=handle_benchmark_selection,
            inputs=[benchmark_filter],
            outputs=[benchmark_filter]
        ).then(
            fn=update_plot,
            inputs=[metric_radio, benchmark_filter, classifier_filter, current_data],
            outputs=[comparison_plot]
        )
        
        classifier_filter.change(
            fn=handle_classifier_selection,
            inputs=[classifier_filter],
            outputs=[classifier_filter]
        ).then(
            fn=update_plot,
            inputs=[metric_radio, benchmark_filter, classifier_filter, current_data],
            outputs=[comparison_plot]
        )
        
        refresh_btn.click(
            fn=update_plot,
            inputs=[metric_radio, benchmark_filter, classifier_filter, current_data],
            outputs=[comparison_plot]
        )
        
        if dclm_classifier:
            def inference_wrapper(text, data_state, metric, bench_filter, clf_filter):
                _, benchmark, _, _, dataset_name = data_state
                return perform_inference(text, benchmark, metric, bench_filter, clf_filter, dataset_name, dclm_classifier)
            
            inference_input.change(
                fn=inference_wrapper,
                inputs=[inference_input, current_data, metric_radio, benchmark_filter, classifier_filter],
                outputs=[comparison_plot]
            )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
