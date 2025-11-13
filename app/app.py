"""
Benchmark in a Haystack - Interactive Visualization App
Explore and analyze how quality classifiers rank benchmark samples
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import json
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = Path("../cache")
RESULTS_DIR = Path("../results")
COLOR_PALETTE = px.colors.qualitative.Set2

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_cache_files() -> Dict[str, pd.DataFrame]:
    """Load all classifier cache JSON files and return as DataFrames."""
    cache_files = list(CACHE_DIR.glob("*Classifier.json"))
    
    if not cache_files:
        print(f"⚠️  No cache files found in {CACHE_DIR}")
        return {}
    
    classifiers_data = {}
    
    for cache_file in cache_files:
        classifier_name = cache_file.stem  # e.g., "DCLMClassifier"
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Convert nested dict to DataFrame
            records = []
            for doc_hash, doc_data in data.items():
                record = {
                    'doc_hash': doc_hash,
                    'classifier': classifier_name,
                    **doc_data
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            classifiers_data[classifier_name] = df
            print(f"✓ Loaded {classifier_name}: {len(df)} documents")
            
        except Exception as e:
            print(f"✗ Error loading {cache_file}: {e}")
    
    return classifiers_data


def build_combined_dataframe(classifiers_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine all classifier DataFrames into a single unified DataFrame."""
    if not classifiers_data:
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    combined = pd.concat(classifiers_data.values(), ignore_index=True)
    
    # Ensure proper data types
    if 'score' in combined.columns:
        combined['score'] = pd.to_numeric(combined['score'], errors='coerce')
    if 'benchmark_index' in combined.columns:
        combined['benchmark_index'] = pd.to_numeric(combined['benchmark_index'], errors='coerce')
    
    print(f"✓ Combined dataset: {len(combined)} total records")
    print(f"  - Classifiers: {combined['classifier'].nunique()}")
    print(f"  - Documents: {combined['id'].nunique()}")
    print(f"  - Benchmarks: {combined[combined['contains_benchmark'] == True]['id'].nunique()}")
    
    return combined


def compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ranks and percentiles for each classifier."""
    df = df.copy()
    
    # Compute ranks within each classifier (lower rank = higher score)
    df['rank'] = df.groupby('classifier')['score'].rank(ascending=False, method='min')
    
    # Compute percentiles
    df['percentile'] = df.groupby('classifier')['rank'].transform(
        lambda x: (x.max() - x + 1) / x.max() * 100
    )
    
    return df


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare all data for visualization."""
    classifiers_data = load_cache_files()
    
    if not classifiers_data:
        return pd.DataFrame(), pd.DataFrame()
    
    combined_df = build_combined_dataframe(classifiers_data)
    combined_df = compute_ranks(combined_df)
    
    # Separate benchmark-only data
    benchmark_df = combined_df[combined_df['contains_benchmark'] == True].copy()
    
    return combined_df, benchmark_df


# ============================================================================
# VISUALIZATION FUNCTIONS - TAB 1: CLASSIFIER COMPARISON
# ============================================================================

def plot_classifier_comparison(benchmark_df: pd.DataFrame, 
                              selected_benchmarks: List[str] = None,
                              selected_classifiers: List[str] = None,
                              metric: str = "rank") -> go.Figure:
    """Create interactive strip plot comparing classifier rankings."""
    if benchmark_df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    df = benchmark_df.copy()
    
    # Apply filters
    if selected_benchmarks and "All" not in selected_benchmarks:
        df = df[df['benchmark_type'].isin(selected_benchmarks)]
    if selected_classifiers and "All" not in selected_classifiers:
        df = df[df['classifier'].isin(selected_classifiers)]
    
    if df.empty:
        return go.Figure().add_annotation(text="No data matching filters", showarrow=False)
    
    # Choose metric
    y_column = metric
    y_label = "Rank (lower is better)" if metric == "rank" else "Percentile (higher is better)"
    
    # Create figure
    fig = px.strip(
        df,
        x='classifier',
        y=y_column,
        color='benchmark_type',
        hover_data=['id', 'score', 'rank', 'percentile'],
        title=f"Benchmark Rankings Across Classifiers ({metric.capitalize()})",
        color_discrete_sequence=COLOR_PALETTE
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white')))
    fig.update_layout(
        xaxis_title="Classifier",
        yaxis_title=y_label,
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(title="Benchmark Type")
    )
    
    if metric == "rank":
        fig.update_yaxes(autorange="reversed")
    
    return fig


def create_summary_stats_table(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics table for classifiers."""
    if benchmark_df.empty:
        return pd.DataFrame()
    
    stats = benchmark_df.groupby('classifier').agg({
        'rank': ['mean', 'median', 'min', 'max'],
        'percentile': ['mean', 'median'],
        'score': ['mean', 'median']
    }).round(2)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.reset_index()
    
    # Rename for readability
    stats.columns = [
        'Classifier', 'Mean Rank', 'Median Rank', 'Best Rank', 'Worst Rank',
        'Mean Percentile', 'Median Percentile', 'Mean Score', 'Median Score'
    ]
    
    # Sort by mean rank (lower is better)
    stats = stats.sort_values('Mean Rank')
    
    return stats


# ============================================================================
# VISUALIZATION FUNCTIONS - TAB 2: SCORE DISTRIBUTIONS
# ============================================================================

def plot_score_distributions(combined_df: pd.DataFrame, 
                            selected_classifiers: List[str] = None,
                            plot_type: str = "violin") -> go.Figure:
    """Create distribution plots for classifier scores."""
    if combined_df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    df = combined_df.copy()
    
    # Apply filters
    if selected_classifiers and "All" not in selected_classifiers:
        df = df[df['classifier'].isin(selected_classifiers)]
    
    if df.empty:
        return go.Figure().add_annotation(text="No data matching filters", showarrow=False)
    
    if plot_type == "violin":
        fig = px.violin(
            df,
            x='classifier',
            y='score',
            color='classifier',
            box=True,
            points='outliers',
            title="Score Distributions by Classifier (Violin Plot)",
            color_discrete_sequence=COLOR_PALETTE
        )
    elif plot_type == "box":
        fig = px.box(
            df,
            x='classifier',
            y='score',
            color='classifier',
            points='outliers',
            title="Score Distributions by Classifier (Box Plot)",
            color_discrete_sequence=COLOR_PALETTE
        )
    else:  # histogram
        fig = px.histogram(
            df,
            x='score',
            color='classifier',
            marginal='rug',
            barmode='overlay',
            opacity=0.6,
            title="Score Distributions by Classifier (Histogram)",
            color_discrete_sequence=COLOR_PALETTE
        )
    
    fig.update_layout(
        xaxis_title="Classifier" if plot_type != "histogram" else "Score",
        yaxis_title="Score" if plot_type != "histogram" else "Count",
        showlegend=True,
        height=600
    )
    
    return fig


def plot_benchmark_in_distribution(combined_df: pd.DataFrame, 
                                   benchmark_df: pd.DataFrame,
                                   classifier: str) -> go.Figure:
    """Show where benchmarks fall in the overall score distribution."""
    if combined_df.empty or benchmark_df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    clf_all = combined_df[combined_df['classifier'] == classifier]
    clf_bench = benchmark_df[benchmark_df['classifier'] == classifier]
    
    if clf_all.empty:
        return go.Figure().add_annotation(text=f"No data for {classifier}", showarrow=False)
    
    fig = go.Figure()
    
    # All documents histogram
    fig.add_trace(go.Histogram(
        x=clf_all['score'],
        name='All Documents',
        opacity=0.6,
        nbinsx=50
    ))
    
    # Benchmark markers
    if not clf_bench.empty:
        for bench_type in clf_bench['benchmark_type'].unique():
            bench_data = clf_bench[clf_bench['benchmark_type'] == bench_type]
            fig.add_trace(go.Scatter(
                x=bench_data['score'],
                y=[0] * len(bench_data),
                mode='markers',
                name=f'{bench_type} benchmarks',
                marker=dict(size=12, symbol='diamond', line=dict(width=2, color='white')),
                text=bench_data['id'],
                hovertemplate='%{text}<br>Score: %{x:.3f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=f"Benchmark Scores in {classifier} Distribution",
        xaxis_title="Score",
        yaxis_title="Count",
        showlegend=True,
        height=500,
        barmode='overlay'
    )
    
    return fig


# ============================================================================
# VISUALIZATION FUNCTIONS - TAB 3: BENCHMARK ANALYSIS
# ============================================================================

def compute_rank_correlation(benchmark_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rank correlation matrices between classifiers."""
    if benchmark_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Pivot to get ranks for each classifier
    pivot_df = benchmark_df.pivot_table(
        index='id',
        columns='classifier',
        values='rank'
    )
    
    # Compute correlations
    spearman_corr = pivot_df.corr(method='spearman')
    
    # Compute Kendall's tau
    kendall_corr = pd.DataFrame(
        index=pivot_df.columns,
        columns=pivot_df.columns,
        dtype=float
    )
    
    for col1 in pivot_df.columns:
        for col2 in pivot_df.columns:
            if col1 == col2:
                kendall_corr.loc[col1, col2] = 1.0
            else:
                # Remove NaN pairs
                mask = pivot_df[[col1, col2]].notna().all(axis=1)
                if mask.sum() > 0:
                    tau, _ = kendalltau(
                        pivot_df.loc[mask, col1],
                        pivot_df.loc[mask, col2]
                    )
                    kendall_corr.loc[col1, col2] = tau
    
    return spearman_corr, kendall_corr.astype(float)


def plot_correlation_heatmap(benchmark_df: pd.DataFrame, method: str = "spearman") -> go.Figure:
    """Create correlation heatmap between classifiers."""
    if benchmark_df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    spearman_corr, kendall_corr = compute_rank_correlation(benchmark_df)
    
    if spearman_corr.empty:
        return go.Figure().add_annotation(text="Insufficient data for correlation", showarrow=False)
    
    corr_matrix = spearman_corr if method == "spearman" else kendall_corr
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=f"Classifier Rank Agreement ({method.capitalize()}'s Correlation)",
        xaxis_title="Classifier",
        yaxis_title="Classifier",
        height=600,
        width=700
    )
    
    return fig


def create_benchmark_detail_table(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Create detailed table showing each benchmark's performance across classifiers."""
    if benchmark_df.empty:
        return pd.DataFrame()
    
    # Pivot to get all classifier scores for each benchmark
    pivot_df = benchmark_df.pivot_table(
        index=['id', 'benchmark_type', 'benchmark_index'],
        columns='classifier',
        values=['rank', 'percentile', 'score']
    )
    
    # Flatten multi-index columns
    pivot_df.columns = ['_'.join(map(str, col)).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df.reset_index()
    
    return pivot_df


def plot_best_worst_benchmarks(benchmark_df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Plot best and worst performing benchmarks across classifiers."""
    if benchmark_df.empty:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    # Compute average percentile per benchmark
    avg_perf = benchmark_df.groupby('id').agg({
        'percentile': 'mean',
        'benchmark_type': 'first'
    }).reset_index()
    
    # Get top and bottom n
    top_n = avg_perf.nlargest(n, 'percentile')
    bottom_n = avg_perf.nsmallest(n, 'percentile')
    
    # Combine
    selected = pd.concat([top_n, bottom_n])
    selected['category'] = ['Top Performers'] * len(top_n) + ['Bottom Performers'] * len(bottom_n)
    
    fig = px.bar(
        selected.sort_values('percentile'),
        y='id',
        x='percentile',
        color='benchmark_type',
        facet_col='category',
        title=f"Top & Bottom {n} Benchmarks by Average Percentile",
        color_discrete_sequence=COLOR_PALETTE,
        orientation='h'
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis_title="Average Percentile"
    )
    
    return fig


# ============================================================================
# VISUALIZATION FUNCTIONS - TAB 4: DOCUMENT EXPLORER
# ============================================================================

def create_searchable_table(combined_df: pd.DataFrame,
                           source_filter: List[str] = None,
                           benchmark_filter: List[str] = None,
                           min_score: float = None,
                           max_score: float = None) -> pd.DataFrame:
    """Create filtered and formatted table for document exploration."""
    if combined_df.empty:
        return pd.DataFrame()
    
    df = combined_df.copy()
    
    # Apply filters
    if source_filter and "All" not in source_filter:
        df = df[df['source'].isin(source_filter)]
    
    if benchmark_filter and "All" not in benchmark_filter:
        if "Non-benchmark" in benchmark_filter:
            df = df[df['contains_benchmark'] == False]
        else:
            df = df[df['benchmark_type'].isin(benchmark_filter)]
    
    if min_score is not None:
        df = df[df['score'] >= min_score]
    if max_score is not None:
        df = df[df['score'] <= max_score]
    
    # Select and format columns
    display_columns = ['id', 'classifier', 'source', 'score', 'rank', 'percentile', 
                       'contains_benchmark', 'benchmark_type', 'benchmark_index']
    
    # Keep only existing columns
    display_columns = [col for col in display_columns if col in df.columns]
    df = df[display_columns]
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(3)
    
    return df


# ============================================================================
# VISUALIZATION FUNCTIONS - TAB 5: STATISTICS & INSIGHTS
# ============================================================================

def compute_agreement_statistics(benchmark_df: pd.DataFrame) -> Dict:
    """Compute various agreement statistics between classifiers."""
    if benchmark_df.empty:
        return {}
    
    spearman_corr, kendall_corr = compute_rank_correlation(benchmark_df)
    
    if spearman_corr.empty:
        return {"error": "Insufficient data"}
    
    # Get upper triangle values (excluding diagonal)
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool), k=1)
    spearman_values = spearman_corr.values[mask]
    kendall_values = kendall_corr.values[mask]
    
    stats = {
        "spearman": {
            "mean": float(np.mean(spearman_values)),
            "median": float(np.median(spearman_values)),
            "min": float(np.min(spearman_values)),
            "max": float(np.max(spearman_values)),
            "std": float(np.std(spearman_values))
        },
        "kendall": {
            "mean": float(np.mean(kendall_values)),
            "median": float(np.median(kendall_values)),
            "min": float(np.min(kendall_values)),
            "max": float(np.max(kendall_values)),
            "std": float(np.std(kendall_values))
        }
    }
    
    return stats


def detect_outliers(benchmark_df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """Detect benchmarks with unusual rankings (outliers)."""
    if benchmark_df.empty:
        return pd.DataFrame()
    
    # For each benchmark, compute variability in percentile ranks
    outliers = []
    
    for bench_id in benchmark_df['id'].unique():
        bench_data = benchmark_df[benchmark_df['id'] == bench_id]
        
        if len(bench_data) < 2:
            continue
        
        percentiles = bench_data['percentile'].values
        std = np.std(percentiles)
        mean = np.mean(percentiles)
        cv = std / mean if mean > 0 else 0  # Coefficient of variation
        
        # High coefficient of variation indicates inconsistent rankings
        if cv > threshold:
            outliers.append({
                'id': bench_id,
                'benchmark_type': bench_data['benchmark_type'].iloc[0],
                'mean_percentile': mean,
                'std_percentile': std,
                'cv': cv,
                'min_percentile': percentiles.min(),
                'max_percentile': percentiles.max(),
                'range': percentiles.max() - percentiles.min()
            })
    
    if outliers:
        outliers_df = pd.DataFrame(outliers).sort_values('cv', ascending=False)
        return outliers_df
    
    return pd.DataFrame()


def generate_insights_report(combined_df: pd.DataFrame, 
                            benchmark_df: pd.DataFrame) -> str:
    """Generate a text summary of key insights."""
    if combined_df.empty or benchmark_df.empty:
        return "No data available for analysis."
    
    report = []
    report.append("# Benchmark-in-a-Haystack Analysis Summary\n")
    
    # Dataset overview
    report.append("## Dataset Overview")
    report.append(f"- Total documents: {combined_df['id'].nunique():,}")
    report.append(f"- Benchmark documents: {benchmark_df['id'].nunique():,}")
    report.append(f"- Classifiers: {combined_df['classifier'].nunique()}")
    report.append(f"- Benchmark types: {', '.join(benchmark_df['benchmark_type'].unique())}\n")
    
    # Classifier performance
    report.append("## Classifier Performance")
    summary_stats = create_summary_stats_table(benchmark_df)
    if not summary_stats.empty:
        best_classifier = summary_stats.iloc[0]['Classifier']
        best_rank = summary_stats.iloc[0]['Mean Rank']
        report.append(f"- Best performing classifier: **{best_classifier}** (mean rank: {best_rank:.1f})")
        
        worst_classifier = summary_stats.iloc[-1]['Classifier']
        worst_rank = summary_stats.iloc[-1]['Mean Rank']
        report.append(f"- Worst performing classifier: **{worst_classifier}** (mean rank: {worst_rank:.1f})\n")
    
    # Agreement between classifiers
    report.append("## Classifier Agreement")
    stats = compute_agreement_statistics(benchmark_df)
    if stats and "error" not in stats:
        report.append(f"- Mean Spearman correlation: **{stats['spearman']['mean']:.3f}**")
        report.append(f"- Mean Kendall's tau: **{stats['kendall']['mean']:.3f}**")
        
        if stats['spearman']['mean'] > 0.7:
            report.append("- ✓ High agreement between classifiers")
        elif stats['spearman']['mean'] > 0.4:
            report.append("- ⚠ Moderate agreement between classifiers")
        else:
            report.append("- ✗ Low agreement between classifiers - significant divergence in rankings")
        report.append("")
    
    # Outliers
    report.append("## Ranking Inconsistencies")
    outliers_df = detect_outliers(benchmark_df, threshold=0.5)
    if not outliers_df.empty:
        report.append(f"- Found {len(outliers_df)} benchmarks with highly variable rankings")
        report.append("- Top 3 most inconsistent:")
        for i, row in outliers_df.head(3).iterrows():
            report.append(f"  - {row['id']}: percentile range {row['min_percentile']:.1f}% - {row['max_percentile']:.1f}%")
    else:
        report.append("- No significant ranking inconsistencies detected")
    report.append("")
    
    # Benchmark type analysis
    report.append("## Benchmark Type Performance")
    for bench_type in benchmark_df['benchmark_type'].unique():
        bench_subset = benchmark_df[benchmark_df['benchmark_type'] == bench_type]
        avg_percentile = bench_subset['percentile'].mean()
        report.append(f"- {bench_type}: {avg_percentile:.1f}% average percentile")
    
    return "\n".join(report)


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_app():
    """Create and configure the Gradio interface."""
    
    # Load data once at startup
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    combined_df, benchmark_df = load_all_data()
    
    if combined_df.empty:
        print("\n⚠️  No data loaded. Please check cache directory.")
        print(f"Expected location: {CACHE_DIR.absolute()}")
        
        # Create minimal app with error message
        with gr.Blocks(theme=gr.themes.Soft()) as app:
            gr.Markdown("# ⚠️ No Data Found")
            gr.Markdown(f"""
            Could not load cache files from `{CACHE_DIR.absolute()}`
            
            Please ensure:
            1. Cache files exist in the `cache/` directory
            2. Files follow the naming pattern `*Classifier.json`
            3. JSON files contain valid scoring results
            """)
        return app
    
    # Extract filter options
    classifiers = sorted(combined_df['classifier'].unique().tolist())
    benchmark_types = sorted(benchmark_df['benchmark_type'].unique().tolist())
    sources = sorted(combined_df['source'].unique().tolist())
    
    print("\n✓ Data loaded successfully!")
    print("="*60 + "\n")
    
    # Create Gradio interface
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Benchmark in a Haystack - Visualization",
        css=".gradio-container {max-width: 1400px !important}"
    ) as app:
        
        gr.Markdown("""
        # 📊 Benchmark in a Haystack - Interactive Visualization
        
        Explore how different quality classifiers rank benchmark samples when inserted into a document corpus.
        """)
        
        # ====================================================================
        # TAB 1: CLASSIFIER COMPARISON
        # ====================================================================
        with gr.Tab("🎯 Classifier Comparison"):
            gr.Markdown("""
            ### Compare Benchmark Rankings Across Classifiers
            See how different classifiers rank the same benchmark samples.
            Lower ranks and higher percentiles indicate the classifier considers the benchmark high-quality.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    metric_radio = gr.Radio(
                        choices=["rank", "percentile"],
                        value="rank",
                        label="Display Metric",
                        info="Choose between absolute rank or percentile"
                    )
                    benchmark_filter_1 = gr.CheckboxGroup(
                        choices=["All"] + benchmark_types,
                        value=["All"],
                        label="Benchmark Types",
                        info="Filter by benchmark type"
                    )
                    classifier_filter_1 = gr.CheckboxGroup(
                        choices=["All"] + classifiers,
                        value=["All"],
                        label="Classifiers",
                        info="Select classifiers to display"
                    )
                    refresh_btn_1 = gr.Button("🔄 Refresh Plot", variant="primary")
                
                with gr.Column(scale=3):
                    comparison_plot = gr.Plot(label="Classifier Comparison")
            
            gr.Markdown("### Summary Statistics")
            summary_table = gr.Dataframe(
                value=create_summary_stats_table(benchmark_df),
                label="Performance Summary by Classifier",
                interactive=False
            )
            
            # Update function
            def update_comparison(metric, bench_filter, clf_filter):
                fig = plot_classifier_comparison(benchmark_df, bench_filter, clf_filter, metric)
                return fig
            
            refresh_btn_1.click(
                fn=update_comparison,
                inputs=[metric_radio, benchmark_filter_1, classifier_filter_1],
                outputs=[comparison_plot]
            )
            
            # Initial load
            comparison_plot.value = plot_classifier_comparison(benchmark_df, metric="rank")
        
        # ====================================================================
        # TAB 2: SCORE DISTRIBUTIONS
        # ====================================================================
        with gr.Tab("📈 Score Distributions"):
            gr.Markdown("""
            ### Visualize Score Distributions
            Explore how scores are distributed for each classifier.
            See where benchmarks fall within the overall distribution.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    plot_type_radio = gr.Radio(
                        choices=["violin", "box", "histogram"],
                        value="violin",
                        label="Plot Type"
                    )
                    classifier_filter_2 = gr.CheckboxGroup(
                        choices=["All"] + classifiers,
                        value=["All"],
                        label="Classifiers"
                    )
                    refresh_btn_2 = gr.Button("🔄 Refresh Plot", variant="primary")
                
                with gr.Column(scale=3):
                    distribution_plot = gr.Plot(label="Score Distributions")
            
            gr.Markdown("### Benchmark Position in Distribution")
            gr.Markdown("See where benchmark samples fall in a specific classifier's distribution.")
            
            with gr.Row():
                classifier_select = gr.Dropdown(
                    choices=classifiers,
                    value=classifiers[0] if classifiers else None,
                    label="Select Classifier"
                )
                refresh_btn_3 = gr.Button("🔄 Show Distribution", variant="secondary")
            
            benchmark_dist_plot = gr.Plot(label="Benchmarks in Distribution")
            
            # Update functions
            def update_distributions(plot_type, clf_filter):
                return plot_score_distributions(combined_df, clf_filter, plot_type)
            
            def update_benchmark_dist(classifier):
                return plot_benchmark_in_distribution(combined_df, benchmark_df, classifier)
            
            refresh_btn_2.click(
                fn=update_distributions,
                inputs=[plot_type_radio, classifier_filter_2],
                outputs=[distribution_plot]
            )
            
            refresh_btn_3.click(
                fn=update_benchmark_dist,
                inputs=[classifier_select],
                outputs=[benchmark_dist_plot]
            )
            
            # Initial load
            distribution_plot.value = plot_score_distributions(combined_df, plot_type="violin")
            if classifiers:
                benchmark_dist_plot.value = plot_benchmark_in_distribution(
                    combined_df, benchmark_df, classifiers[0]
                )
        
        # ====================================================================
        # TAB 3: BENCHMARK ANALYSIS
        # ====================================================================
        with gr.Tab("🔬 Benchmark Analysis"):
            gr.Markdown("""
            ### Deep Dive into Benchmark Performance
            Analyze correlations between classifiers and identify best/worst performing benchmarks.
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Classifier Agreement Heatmap")
                    corr_method = gr.Radio(
                        choices=["spearman", "kendall"],
                        value="spearman",
                        label="Correlation Method"
                    )
                    refresh_btn_4 = gr.Button("🔄 Update Heatmap", variant="primary")
                    heatmap_plot = gr.Plot(label="Correlation Heatmap")
                
                with gr.Column():
                    gr.Markdown("#### Best & Worst Performers")
                    n_benchmarks = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of benchmarks to show"
                    )
                    refresh_btn_5 = gr.Button("🔄 Update Chart", variant="primary")
                    best_worst_plot = gr.Plot(label="Top & Bottom Benchmarks")
            
            gr.Markdown("### Detailed Benchmark Rankings")
            detail_table = gr.Dataframe(
                value=create_benchmark_detail_table(benchmark_df).head(50),
                label="Benchmark Details (showing first 50)",
                interactive=False,
                wrap=True
            )
            
            # Update functions
            def update_heatmap(method):
                return plot_correlation_heatmap(benchmark_df, method)
            
            def update_best_worst(n):
                return plot_best_worst_benchmarks(benchmark_df, int(n))
            
            refresh_btn_4.click(
                fn=update_heatmap,
                inputs=[corr_method],
                outputs=[heatmap_plot]
            )
            
            refresh_btn_5.click(
                fn=update_best_worst,
                inputs=[n_benchmarks],
                outputs=[best_worst_plot]
            )
            
            # Initial load
            heatmap_plot.value = plot_correlation_heatmap(benchmark_df, "spearman")
            best_worst_plot.value = plot_best_worst_benchmarks(benchmark_df, 10)
        
        # ====================================================================
        # TAB 4: DOCUMENT EXPLORER
        # ====================================================================
        with gr.Tab("🔍 Document Explorer"):
            gr.Markdown("""
            ### Explore and Filter Documents
            Search, filter, and export the complete dataset.
            """)
            
            with gr.Row():
                source_filter = gr.CheckboxGroup(
                    choices=["All"] + sources,
                    value=["All"],
                    label="Source Filter"
                )
                benchmark_filter_4 = gr.CheckboxGroup(
                    choices=["All", "Non-benchmark"] + benchmark_types,
                    value=["All"],
                    label="Benchmark Filter"
                )
            
            with gr.Row():
                min_score_slider = gr.Slider(
                    minimum=float(combined_df['score'].min()),
                    maximum=float(combined_df['score'].max()),
                    value=float(combined_df['score'].min()),
                    label="Minimum Score"
                )
                max_score_slider = gr.Slider(
                    minimum=float(combined_df['score'].min()),
                    maximum=float(combined_df['score'].max()),
                    value=float(combined_df['score'].max()),
                    label="Maximum Score"
                )
            
            apply_filter_btn = gr.Button("🔍 Apply Filters", variant="primary")
            
            explorer_table = gr.Dataframe(
                value=create_searchable_table(combined_df).head(100),
                label="Documents (showing first 100 after filtering)",
                interactive=False,
                wrap=True
            )
            
            # Update function
            def update_explorer(src_filter, bench_filter, min_score, max_score):
                filtered = create_searchable_table(
                    combined_df, src_filter, bench_filter, min_score, max_score
                )
                return filtered.head(100)
            
            apply_filter_btn.click(
                fn=update_explorer,
                inputs=[source_filter, benchmark_filter_4, min_score_slider, max_score_slider],
                outputs=[explorer_table]
            )
        
        # ====================================================================
        # TAB 5: STATISTICS & INSIGHTS
        # ====================================================================
        with gr.Tab("📊 Statistics & Insights"):
            gr.Markdown("""
            ### Statistical Analysis and Key Findings
            Quantitative metrics and automated insights about classifier agreement and benchmark performance.
            """)
            
            gr.Markdown("#### Automated Analysis Report")
            insights_text = gr.Markdown(
                value=generate_insights_report(combined_df, benchmark_df)
            )
            
            gr.Markdown("#### Ranking Inconsistencies (Outliers)")
            gr.Markdown("""
            Benchmarks with high variability in percentile rankings across classifiers.
            High coefficient of variation (CV) indicates strong disagreement.
            """)
            
            outliers_table = gr.Dataframe(
                value=detect_outliers(benchmark_df, threshold=0.5),
                label="Inconsistent Rankings",
                interactive=False
            )
            
            gr.Markdown("#### Correlation Matrices")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Spearman Correlation**")
                    spearman_corr, _ = compute_rank_correlation(benchmark_df)
                    spearman_df = gr.Dataframe(
                        value=spearman_corr.round(3),
                        label="Spearman Rank Correlation",
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("**Kendall's Tau**")
                    _, kendall_corr = compute_rank_correlation(benchmark_df)
                    kendall_df = gr.Dataframe(
                        value=kendall_corr.round(3),
                        label="Kendall's Tau",
                        interactive=False
                    )
        
        # Footer
        gr.Markdown("""
        ---
        **Benchmark in a Haystack** | Visualizing quality classifier performance on benchmark samples
        """)
    
    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

