<div align="center">
  <img src="biahs-banner.png" alt="Benchmark in a Haystack Banner">
</div>

Quality filters decide what enters a pretraining corpus. If they rank benchmark
questions above ordinary web text, then the curation pipeline is itself a
contamination channel.

Insert benchmark items (MMLU, GSM8K, GPQA, ARC, HellaSwag, PIQA, TruthfulQA)
into a FineWeb corpus, score every document with six quality classifiers, and
read where the benchmark items land in the ranking.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run experiment:
```bash
python haystack.py --config config.yaml
```

If you want to download models first for offline use:
```bash
python haystack.py --download-models
```

## Configuration

Edit `config.yaml` to configure:

- `num_docs`: Number of documents (default: 100000)
- `inject_inside`: true = inject benchmarks into docs, false = separate docs (default: false)
- `prefilter_hq`: Use only high-quality FineWeb documents (default: false)
- `min_hq_score`: Minimum quality score threshold (default: 0.7)
- `benchmarks`: Configure count and subjects per benchmark
- `classifiers`: Enable/disable classifiers and set batch sizes

## Output

Results saved to `results/TIMESTAMP/`:
- `benchmark_ranks_all_classifiers.json`: Rankings for all classifiers
- `benchmark_ranks_by_classifier.png`: Visual comparison
- `benchmark_percentiles_by_classifier.png`: Normalized view

## Classifiers

- DCLMClassifier
- FinewebEduClassifier
- GaperonClassifier
- NemoCuratorEduClassifier
- EuroFilterClassifier
- TextbookFastTextClassifier
- FinePDFsEduClassifier
- FinePDFsEduClassifierV2
- FinePDFsDCLMClassifier

## Adding Benchmarks

To add a new benchmark, edit `benchmarks.py`:

1. **Create a class** that inherits from `Benchmark` ABC

2. **Define class attributes** (optional but recommended):
   - `dataset`: HuggingFace dataset name (e.g., `"cais/mmlu"`)
   - `split`: Dataset split to use (e.g., `"test"`, `"validation"`)
   - `config` or `name`: Dataset configuration if needed
   - `format_template`: String template for formatting samples

3. **Implement required methods**:

   - `load_samples(self, count=5, subjects=None)`: Load samples from the dataset
     - **Returns**: List of dicts with keys:
       - `"data"`: The raw sample from the dataset
       - `"benchmark_type"`: String identifier for your benchmark
       - `"subject"` (optional): Subject name if applicable
     - Use `random.sample()` to select random samples if needed
     - Handle `subjects` parameter if your benchmark has categories (like MMLU)

   - `format_sample(self, sample, subject=None)`: Convert a sample to text
     - **Parameters**: 
       - `sample`: Dict from `load_samples()` with `"data"` key
       - `subject`: Optional subject name
     - **Returns**: Formatted string ready for insertion into corpus
     - Use `format_template.format()` for consistent formatting

4. **Register** your benchmark in the `BENCHMARKS` dict at the bottom of the file:
   ```python
   BENCHMARKS = {
       "your_benchmark": YourBenchmark(),
       ...
   }
   ```

**Example**: See `GSM8KBenchmark` for a simple benchmark or `MMLUBenchmark` for one with subject categories.

## Adding Classifiers

To add a new classifier, edit `models.py` and choose the appropriate base class:

### Option 1: FastText-based Classifier (like DCLMClassifier)

Inherit from `DocumentClassifier` and implement:

- `__init__(self, classifier_config=None)`: Initialize your model
- `_score_documents_impl(self, documents)`: Score documents and return results list
- `download_model(models_dir="models")`: Static method to download model files

### Option 2: Transformer-based Classifier (like FinewebEduClassifier)

Inherit from `TransformerClassifier` and implement:

- `get_model_config(self)`: Return dict with `model_dir`, `hub_name`, `trust_remote_code` (optional), `max_length` (optional), `torch_dtype` (optional)
- `process_outputs(self, outputs, doc_batch)`: Process model outputs into results list with keys: `id`, `source`, `contains_benchmark`, `benchmark_type`, `benchmark_index`, `score`
- `_process_inputs(self, inputs)` (optional): Modify inputs before passing to model

After implementing your classifier, add it to the `classifiers` section in `config.yaml`.

## Citation

Based on methodology from:
```
@misc{godey2025gaperonpepperedenglishfrenchgenerative,
      title={Gaperon: A Peppered English-French Generative Language Model Suite}, 
      author={Nathan Godey and Wissam Antoun and Rian Touchent and Rachel Bawden and Éric de la Clergerie and Benoît Sagot and Djamé Seddah},
      year={2025},
      eprint={2510.25771},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.25771}, 
}
```

## License

MIT