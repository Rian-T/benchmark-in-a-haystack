# Benchmark in a Haystack

Evaluate how quality filters rank benchmark samples. Insert benchmark items (MMLU, GSM8K, GPQA) into a corpus and measure their ranking by different quality classifiers.

## Citation

Based on methodology from:
```
Godey, N., Antoun, W., Touchent, R., Bawden, R., de la Clergerie, É., Sagot, B., & Seddah, D. (2025).
Gaperon: A Peppered English-French Generative Language Model Suite.
arXiv preprint arXiv:2510.25771.
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Download models first:
```bash
python haystack.py --download-models
```

Run experiment:
```bash
python haystack.py --num-docs 100000
```

## Options

- `--num-docs N`: Number of documents (default: 100000)
- `--mmlu-count N`: Samples per MMLU subject (default: 3)
- `--mmlu-subjects`: Comma-separated subjects (default: anatomy,computer_security,high_school_geography,moral_scenarios,college_physics)
- `--separate`: Create separate documents for benchmarks instead of injecting
- `--prefilter-hq`: Use only high-quality FineWeb documents
- `--min-hq-score`: Minimum quality score threshold (default: 0.7)

## Output

Results saved to `results/TIMESTAMP/`:
- `benchmark_ranks_all_classifiers.json`: Rankings for all classifiers
- `benchmark_ranks_by_classifier.png`: Visual comparison
- `benchmark_percentiles_by_classifier.png`: Normalized view

## Classifiers

- **DCLMClassifier**: Instruction/ELI5-style filter
- **FinewebEduClassifier**: Educational value filter
- **TextbookFastTextClassifier**: Textbook quality filter
- **GaperonClassifier**: General quality filter

## Adding Benchmarks

Edit `benchmarks.py`, implement `load_samples` and `format_sample`, register in `BENCHMARKS`.

## Adding Classifiers

Edit `models.py`, inherit from `DocumentClassifier`, implement `score_documents`, add to `CLASSIFIERS` in `haystack.py`.

## License

MIT
