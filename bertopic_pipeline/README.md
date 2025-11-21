# BERTopic Modeling Pipeline

This pipeline performs topic modeling using BERTopic with pre-computed embeddings, HDBSCAN clustering, and UMAP dimensionality reduction.

## Overview

The pipeline consists of 5 main steps:
1. **Load pre-embedded data** from embedding files linked by `embed_id`
2. **Filter documents** based on topic and platform values
3. **Run BERTopic** with HDBSCAN + UMAP using the embeddings
4. **Save outputs** including the topic model and results CSV
5. **Generate LLM labels** using OpenAI API (optional)

## Files

- `bertopic_modeling.py` - Main pipeline script
- `run_analysis.py` - Example usage script for batch processing
- `requirements.txt` - Required Python packages
- `README.md` - This documentation

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up OpenAI API key (optional, for LLM labeling):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Note: The script automatically reads the API key from the `OPENAI_API_KEY` environment variable.

## Usage

### Basic Usage

```bash
python bertopic_modeling.py \
    --input_parquet /path/to/your/documents.parquet \
    --embedding_folder /path/to/embeddings/ \
    --output_path /path/to/output/ \
    --topic "KEYWORDS_COVID19" \
    --platform "twitter"
```

### With LLM Labeling

```bash
python bertopic_modeling.py \
    --input_parquet /path/to/your/documents.parquet \
    --embedding_folder /path/to/embeddings/ \
    --output_path /path/to/output/ \
    --topic "KEYWORDS_COVID19" \
    --platform "twitter" \
    --openai_api_key "your-api-key"
```

### Advanced Options

```bash
python bertopic_modeling.py \
    --input_parquet /path/to/your/documents.parquet \
    --embedding_folder /path/to/embeddings/ \
    --output_path /path/to/output/ \
    --topic "KEYWORDS_COVID19" \
    --platform "twitter" \
    --min_cluster_size 15 \
    --openai_api_key "your-api-key"
```

### Batch Processing

Use the example script to process multiple topic-platform combinations:

```bash
# Edit run_analysis.py to set your paths and configurations
python run_analysis.py
```

## Input Requirements

### Parquet File
Your input parquet file must contain these columns:
- `platform` - Platform identifier (e.g., "twitter", "reddit")
- `topic` - Topic identifier (e.g., "KEYWORDS_COVID19")
- `embed_id` - Unique identifier linking to embeddings
- `post_clean` - Cleaned document text for analysis

### Embedding Files
- Folder containing `.npy` files with embeddings
- Each file should be a dictionary mapping `embed_id` to embedding vectors
- Files are automatically sorted by number in filename

## Outputs

The pipeline generates the following outputs in the specified output directory:

### 1. Topic Model
- `bertopic_model_{topic}_{platform}.pkl` - Serialized BERTopic model

### 2. Topic Results
- `bertopic_results_{topic}_{platform}.csv` - Topic analysis results with:
  - `topic_id` - Topic identifier
  - `topic_size` - Number of documents in topic
  - `representative_words` - Top 10 words (c-TF-IDF)
  - `representative_docs` - 25 representative documents

### 3. LLM-Labeled Results (if API key provided)
- `bertopic_labeled_results_{topic}_{platform}.csv` - Same as above plus:
  - `llm_label` - GPT-generated topic label

## Algorithm Details

### Clustering Configuration
- **UMAP**: 15 neighbors, 5 components, cosine metric
- **HDBSCAN**: Minimum cluster size 10+ (configurable), euclidean metric
- **Vectorizer**: English stop words, 1-2 ngrams, min_df=2

### Representative Document Selection
1. **First document**: Selected using c-TF-IDF (BERTopic default)
2. **Additional documents**: 
   - Random sample up to 200 documents from cluster
   - Diversity sampling for remaining 24 documents (pseudo-MMR)

### Outlier Reduction
- Automatically applied if outliers (topic -1) are found
- Uses probability threshold of 0.1

## Example Configuration

```python
# Example paths for your setup
input_parquet = "/VData/scro4316/ct_prevalence/data/documents.parquet"
embedding_folder = "/VData/scro4316/ct_prevalence/embeddings"
output_path = "/VData/scro4316/ct_prevalence/bertopic_results"
topic = "KEYWORDS_COVID19"
platform = "twitter"
```

## Troubleshooting

### Common Issues

1. **"Insufficient data" error**
   - Increase `min_cluster_size` or ensure more documents match your topic/platform filter
   - Check that `embed_id` values match between parquet and embedding files

2. **"No meaningful topics found"**
   - Try reducing `min_cluster_size`
   - Check document quality and diversity
   - Verify embeddings are loaded correctly

3. **Memory issues with large datasets**
   - Process subsets of topics/platforms separately
   - Reduce embedding dimensions if possible
   - Use a machine with more RAM

4. **LLM labeling fails**
   - Verify OpenAI API key is valid and has credits
   - Check internet connection
   - Try without LLM labeling first

### Performance Tips

- Use SSD storage for faster file I/O
- Increase `min_cluster_size` for faster processing
- Process topic-platform combinations in parallel on different machines
- Pre-filter your parquet file to only include needed data

## Dependencies

See `requirements.txt` for complete list. Key packages:
- `bertopic>=0.15.0`
- `umap-learn>=0.5.3`
- `hdbscan>=0.8.29`
- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `openai>=0.28.0`