# Benchmark in a Haystack - Visualization App

Interactive Gradio application for exploring and analyzing benchmark classifier results.

## Features

✨ **Five Interactive Tabs for Comprehensive Analysis:**

- **🎯 Classifier Comparison**: Compare how different classifiers rank benchmark samples with interactive strip plots
- **📈 Score Distributions**: Visualize score distributions across classifiers with violin/box/histogram plots
- **🔬 Benchmark Analysis**: Deep dive into benchmark performance with correlation heatmaps and detailed tables
- **🔍 Document Explorer**: Search, filter, and explore all documents with their scores across classifiers
- **📊 Statistics & Insights**: Automated statistical analysis and agreement metrics between classifiers

## Installation

```bash
cd app
pip install -r requirements.txt
```

## Quick Start

```bash
python app.py
```

The app will automatically:
- Scan `../cache/` for classifier result JSON files (e.g., `DCLMClassifier.json`)
- Load and combine data from all available classifiers
- Compute rankings and percentiles
- Launch in your browser at `http://localhost:7860`

## Expected Data Format

The app expects cache files in `../cache/` with the following structure:

```json
{
  "doc_hash_1": {
    "id": "doc_id",
    "source": "fineweb",
    "contains_benchmark": false,
    "benchmark_type": null,
    "benchmark_index": null,
    "score": 0.95
  },
  "doc_hash_2": {
    "id": "mmlu_0",
    "source": "mmlu",
    "contains_benchmark": true,
    "benchmark_type": "mmlu",
    "benchmark_index": 0,
    "score": 0.72
  }
}
```

## Data Requirements

The app expects:
- Cache files in `../cache/` with naming pattern `*Classifier.json`
- Each cache file contains document hashes as keys with scoring results
- Optional: Pre-computed analysis results in `../results/`

## Features Overview

### Tab 1: Classifier Comparison
- Interactive strip plots showing benchmark rankings across classifiers
- Percentile comparisons
- Filter by benchmark type or score range
- Summary statistics table

### Tab 2: Score Distributions
- Violin plots of score distributions per classifier
- Overlaid histograms for comparison
- Box plots with statistical summaries
- Benchmark highlighting

### Tab 3: Benchmark Analysis
- Correlation heatmap between classifiers
- Per-benchmark detailed breakdown
- Best/worst performing benchmarks
- Subject-level analysis (e.g., MMLU subjects)

### Tab 4: Document Explorer
- Searchable, sortable data table
- Filter by source, benchmark type, or score
- Export filtered results as CSV
- Detailed document information

### Tab 5: Statistics & Insights
- Kendall's tau and Spearman correlation metrics
- Ranking consistency analysis
- Outlier detection
- Summary insights

## Usage Examples

### Comparing Classifier Performance

1. Navigate to the **🎯 Classifier Comparison** tab
2. Select which benchmark types to display (MMLU, GSM8K, GPQA)
3. Toggle between "rank" and "percentile" views
4. Review the summary statistics table to see which classifier performs best

### Analyzing Score Distributions

1. Go to **📈 Score Distributions** tab
2. Choose visualization type: violin, box, or histogram
3. Select specific classifiers to compare
4. Use the second section to see where benchmarks fall in each classifier's distribution

### Finding Agreement Between Classifiers

1. Open **🔬 Benchmark Analysis** tab
2. View the correlation heatmap to see which classifiers agree most
3. Explore the "Best & Worst Performers" chart
4. Check the detailed table for specific benchmark rankings

### Exploring Individual Documents

1. Visit **🔍 Document Explorer** tab
2. Apply filters by source, benchmark type, or score range
3. Browse the searchable table
4. Export filtered results for further analysis

### Getting Statistical Insights

1. Check **📊 Statistics & Insights** tab
2. Read the automated analysis report
3. Review ranking inconsistencies to find controversial benchmarks
4. Examine correlation matrices for detailed agreement metrics

## Data Summary

After loading, the app displays:
- Total number of documents processed
- Number of classifiers available
- Benchmark types found
- Loading status for each classifier

Example output:
```
✓ Loaded DCLMClassifier: 100,000 documents
✓ Loaded FinewebEduClassifier: 100,000 documents
✓ Loaded GaperonClassifier: 100,000 documents
✓ Combined dataset: 500,000 total records
  - Classifiers: 5
  - Documents: 100,000
  - Benchmarks: 35
```

## Troubleshooting

**No data found:**
- Ensure cache files exist in `../cache/`
- Check that JSON files follow the expected format
- Verify files end with `Classifier.json`

**Visualization issues:**
- Try refreshing the browser
- Check console for error messages
- Ensure all required dependencies are installed

**Performance:**
- Large datasets (>100K documents) may take 10-30 seconds to load
- Use filters to focus on specific data subsets
- Initial plot generation may take a few seconds

## Technical Details

**Data Processing:**
- Loads all cache JSON files on startup
- Computes ranks and percentiles for each classifier
- Caches processed data in memory for fast interaction

**Visualizations:**
- Built with Plotly for interactive, zoomable charts
- All plots support hover tooltips with detailed information
- Export plots as PNG by clicking the camera icon

**Filtering:**
- Real-time filtering without reloading data
- Multiple filter combinations supported
- Filters preserved across tab switches

