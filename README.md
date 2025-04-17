# Haystack Benchmark Ranking Experiment

## Run the Experiment

```bash
python haystack.py --num-docs 100000
```

Options:
- `--separate` : Place benchmarks in separate docs (not injected)
- `--prefilter-hq` : Use only high-quality fineweb docs
- `--min-hq-score 0.7` : Set min high-quality score
- `--fineweb-path <path_or_repo>` : Specify fineweb dataset path or HF repo (default: HuggingFaceFW/fineweb)
- `--download-models` : Download all required models to `models/` and exit

Results (plots, CSV, JSON) are saved in `results/`.

## Download All Models for Offline Use

```bash
python haystack.py --download-models
```

## Add More Benchmarks

1. Edit `benchmarks.py` to add a new benchmark class and register it in `BENCHMARKS`.
2. Implement `load_samples` and `format_sample` methods for your benchmark.

## Add More Models

1. Edit `models.py` and add a new classifier class inheriting from `DocumentClassifier`.
2. Implement the `score_documents(self, documents)` method.
3. Add your classifier to the `CLASSIFIERS` list in `haystack.py`.
