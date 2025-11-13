"""
Demonstration script showing the app's capabilities
Run this to see a summary of what the app can do with your data
"""

import app
import pandas as pd

print("\n" + "="*70)
print("BENCHMARK IN A HAYSTACK - VISUALIZATION APP DEMO")
print("="*70 + "\n")

# Load data
print("📥 Loading data from cache files...")
classifiers_data = app.load_cache_files()

if not classifiers_data:
    print("❌ No cache files found. Please ensure cache files exist in ../cache/")
    exit(1)

combined_df = app.build_combined_dataframe(classifiers_data)
combined_df = app.compute_ranks(combined_df)
benchmark_df = combined_df[combined_df['contains_benchmark'] == True].copy()

print("\n" + "="*70)
print("📊 DATA OVERVIEW")
print("="*70)
print(f"Total records:        {len(combined_df):,}")
print(f"Unique documents:     {combined_df['id'].nunique():,}")
print(f"Benchmark documents:  {benchmark_df['id'].nunique()}")
print(f"Classifiers:          {combined_df['classifier'].nunique()}")
print(f"Benchmark types:      {', '.join(benchmark_df['benchmark_type'].unique())}")

print("\n" + "="*70)
print("🏆 CLASSIFIER PERFORMANCE SUMMARY")
print("="*70)
summary = app.create_summary_stats_table(benchmark_df)
print(summary.to_string(index=False))

print("\n" + "="*70)
print("🤝 CLASSIFIER AGREEMENT")
print("="*70)
stats = app.compute_agreement_statistics(benchmark_df)
if stats and "error" not in stats:
    print(f"Mean Spearman correlation:  {stats['spearman']['mean']:.3f}")
    print(f"Mean Kendall's tau:         {stats['kendall']['mean']:.3f}")
    print(f"Correlation range:          {stats['spearman']['min']:.3f} to {stats['spearman']['max']:.3f}")
    
    if stats['spearman']['mean'] > 0.7:
        print("✅ High agreement - classifiers largely agree on rankings")
    elif stats['spearman']['mean'] > 0.4:
        print("⚠️  Moderate agreement - some divergence in rankings")
    else:
        print("❌ Low agreement - significant divergence in rankings")
else:
    print("⚠️  Insufficient data for correlation analysis")

print("\n" + "="*70)
print("🎯 BENCHMARK TYPE PERFORMANCE")
print("="*70)
for bench_type in sorted(benchmark_df['benchmark_type'].unique()):
    bench_subset = benchmark_df[benchmark_df['benchmark_type'] == bench_type]
    count = bench_subset['id'].nunique()
    avg_percentile = bench_subset['percentile'].mean()
    median_percentile = bench_subset['percentile'].median()
    
    print(f"{bench_type:10s} | Count: {count:3d} | Avg Percentile: {avg_percentile:5.1f}% | Median: {median_percentile:5.1f}%")

print("\n" + "="*70)
print("🔍 RANKING INCONSISTENCIES")
print("="*70)
outliers = app.detect_outliers(benchmark_df, threshold=0.5)
if not outliers.empty:
    print(f"Found {len(outliers)} benchmarks with highly variable rankings")
    print("\nTop 5 most inconsistent benchmarks:")
    for i, row in outliers.head(5).iterrows():
        print(f"  {row['id']:15s} | Range: {row['min_percentile']:5.1f}% - {row['max_percentile']:5.1f}% | CV: {row['cv']:.2f}")
else:
    print("✅ No significant ranking inconsistencies detected")

print("\n" + "="*70)
print("🚀 LAUNCHING GRADIO APP")
print("="*70)
print("\nThe app provides interactive visualizations including:")
print("  • 🎯 Classifier Comparison: Interactive strip plots of rankings")
print("  • 📈 Score Distributions: Violin/box/histogram plots")
print("  • 🔬 Benchmark Analysis: Correlation heatmaps and detailed tables")
print("  • 🔍 Document Explorer: Searchable, filterable data table")
print("  • 📊 Statistics & Insights: Automated analysis and metrics")
print("\nRun 'python app.py' to launch the interactive web interface!")
print("\n" + "="*70 + "\n")

