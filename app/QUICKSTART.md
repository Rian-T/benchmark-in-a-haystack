# Quick Start Guide

## 1. Installation (30 seconds)

```bash
cd app
pip install -r requirements.txt
```

## 2. Run the App (instant)

```bash
python app.py
```

The app will open in your browser at `http://localhost:7860`

## 3. Explore Your Data

The app loads automatically and presents 5 interactive tabs:

### 🎯 Tab 1: Classifier Comparison
**What you'll see:** How each classifier ranks your benchmark samples
- Lower ranks = classifier thinks it's high quality
- Higher percentiles = classifier ranks it highly
- **Try this:** Switch between "rank" and "percentile" views

### 📈 Tab 2: Score Distributions
**What you'll see:** The score distribution for each classifier
- Violin plots show the full distribution shape
- Benchmarks appear as diamonds on the distribution
- **Try this:** Change plot type to see different visualizations

### 🔬 Tab 3: Benchmark Analysis
**What you'll see:** How classifiers agree or disagree
- Heatmap shows correlation between classifiers
- Red = negative correlation, Blue = positive correlation
- **Try this:** Look for high variability benchmarks

### 🔍 Tab 4: Document Explorer
**What you'll see:** All documents with their scores
- Searchable and sortable table
- Filter by source, benchmark type, or score
- **Try this:** Set score filters to find edge cases

### 📊 Tab 5: Statistics & Insights
**What you'll see:** Automated analysis and key findings
- Agreement metrics between classifiers
- Outlier detection
- Summary report
- **Try this:** Check the inconsistencies table for controversial benchmarks

## 4. Understanding the Results

### Good Classifier Performance
- Benchmark ranks in top 10% (rank < 10,000 for 100K docs)
- Percentile > 90%
- Consistently high across all benchmark types

### Classifier Agreement
- Spearman correlation > 0.7 = High agreement
- Spearman correlation 0.4-0.7 = Moderate agreement
- Spearman correlation < 0.4 = Low agreement (interesting!)

### Outliers (Important!)
Benchmarks with high variability in rankings indicate:
- Classifiers measure different quality aspects
- Some benchmarks confuse certain classifiers
- Worth investigating manually!

## 5. Common Questions

**Q: Why are some ranks very different?**
A: Different classifiers optimize for different quality signals. This is expected and interesting!

**Q: Can I export the data?**
A: Yes! Use the Document Explorer tab and apply filters, then copy the table.

**Q: How do I share results?**
A: Take screenshots of the plots (Plotly has a camera icon) or export data from tables.

**Q: The app is slow?**
A: First load takes 10-30 seconds for large datasets. After that, it's instant!

## 6. Pro Tips

1. **Start with Tab 5** (Statistics) to get the big picture
2. **Use Tab 1** to identify best/worst classifiers for your benchmarks
3. **Use Tab 3** to understand classifier agreement
4. **Use Tab 4** to investigate specific documents
5. **Use Tab 2** to understand score distributions

## 7. Need Help?

See the full [README.md](README.md) for:
- Detailed feature descriptions
- Data format requirements
- Troubleshooting guide
- Technical details

## 8. Demo Output

Run `python demo.py` to see a text summary of your data without opening the web interface.

