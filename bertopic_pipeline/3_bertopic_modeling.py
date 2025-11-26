#!/usr/bin/env python3
"""
BERTopic Modeling Pipeline

This script performs topic modeling using BERTopic with the following steps:
1. Load pre-embedded data from embedding files
2. Filter documents by topic and platform
3. Run BERTopic with HDBSCAN + UMAP clustering per topic and platform
4. Save topic model and results
5. Generate LLM-based topic labels

Usage:
    python bertopic_modeling.py --input_parquet <path> --embedding_folder <path> 
                               --output_path <path> --topic <topic> --platform <platform>
"""

import argparse
import os
import sys
import re
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# BERTopic and clustering imports
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# LLM integration
import openai
from bertopic.representation import OpenAI as BertopicOpenAI


def load_embeddings(file_path: str, ids_order: Optional[List] = None) -> Optional[Dict]:
    """
    Load embeddings from a numpy file.
    
    Args:
        file_path (str): Path to the .npy file containing embeddings
        ids_order (List, optional): Order of IDs to load embeddings for
        
    Returns:
        Dict: Dictionary mapping embed_ids to their embeddings
    """
    try:
        embeddings_dict = np.load(file_path, allow_pickle=True).item()
        if ids_order:
            embeddings_ordered = {id_: embeddings_dict[id_] for id_ in ids_order if id_ in embeddings_dict}
            return embeddings_ordered
        return embeddings_dict
    except Exception as e:
        print(f"Failed to load embeddings from {file_path} with error {e}")
        return None


def load_all_embeddings(folder_path: str, ids_order: Optional[List] = None) -> Dict:
    """
    Load all embeddings from a directory containing multiple .npy files.
    
    Args:
        folder_path (str): Path to folder containing embedding files
        ids_order (List, optional): Order of IDs to load embeddings for
        
    Returns:
        Dict: Combined dictionary of all embeddings
    """
    embeddings_dict = {}
    
    if not os.path.exists(folder_path):
        print(f"Embeddings folder {folder_path} does not exist!")
        return embeddings_dict
    
    # Get filenames and sort by number
    try:
        filenames = sorted(
            [f for f in os.listdir(folder_path) if f.endswith('.npy')],
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0
        )
    except Exception as e:
        print(f"Error sorting filenames: {e}")
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    
    for file_name in filenames:
        file_path = os.path.join(folder_path, file_name)
        chunk_embeddings = load_embeddings(file_path, ids_order)
        if chunk_embeddings:
            embeddings_dict.update(chunk_embeddings)
            print(f"Loaded {len(chunk_embeddings)} embeddings from {file_name}")
    
    print(f"Total embeddings loaded: {len(embeddings_dict)}")
    return embeddings_dict


def filter_and_prepare_data(df: pd.DataFrame, topic: str, platform: str, 
                          embeddings_dict: Dict, min_cluster_size: int = 10) -> Tuple[List, List, np.ndarray]:
    """
    Filter documents by topic and platform, and prepare data for BERTopic.
    
    Args:
        df (pd.DataFrame): Input dataframe with columns: platform, topic, embed_id, post_clean
        topic (str): Topic to filter for
        platform (str): Platform to filter for
        embeddings_dict (Dict): Dictionary of embeddings
        min_cluster_size (int): Minimum cluster size for processing
        
    Returns:
        Tuple: (documents, doc_ids, embeddings_array)
    """
    # Filter data by topic and platform
    mask = (df['topic'] == topic) & (df['platform'] == platform)
    subset_df = df[mask].copy()
    
    print(f"Found {len(subset_df)} documents for {topic} on {platform}")
    
    # Filter for documents that have embeddings
    subset_df = subset_df[subset_df['embed_id'].astype(str).isin(embeddings_dict.keys())].copy()
    
    if len(subset_df) < min_cluster_size:
        raise ValueError(f"Insufficient data for {topic} on {platform}: {len(subset_df)} documents")
    
    # Get documents and their embeddings
    doc_ids = subset_df['embed_id'].astype(str).tolist()
    documents = subset_df['post_clean'].tolist()
    
    print(f"Processing {len(documents)} documents after filtering")
    
    # Extract embeddings in order
    doc_embeddings = np.array([embeddings_dict[id_] for id_ in doc_ids])
    
    return documents, doc_ids, doc_embeddings


def create_bertopic_model(documents: List[str], embeddings: np.ndarray, 
                         min_cluster_size: int = 10) -> Tuple[BERTopic, List, np.ndarray]:
    """
    Create and fit BERTopic model with HDBSCAN and UMAP.
    
    Args:
        documents (List[str]): List of documents to cluster
        embeddings (np.ndarray): Document embeddings
        min_cluster_size (int): Minimum cluster size
        
    Returns:
        Tuple: (fitted_topic_model, topics, reduced_embeddings)
    """
    # Initialize UMAP model for dimensionality reduction
    umap_model = UMAP(
        n_neighbors=min(15, len(documents)-1),
        n_components=5,  # Higher dimensions for better clustering
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    # Initialize HDBSCAN for clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(min_cluster_size, 10),
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    # Initialize vectorizer for c-TF-IDF
    # Very conservative parameters for small datasets
    num_docs = len(documents)
    
    # For very small datasets, be extra conservative
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),  # Only unigrams for small datasets
    )
    
    # Initialize BERTopic model
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True
    )
    
    print("Fitting BERTopic model...")
    
    # Fit the model
    topics, probabilities = topic_model.fit_transform(documents, embeddings)
    
    # Check if any topics were found
    topic_info = topic_model.get_topic_info()
    meaningful_topics = len(topic_info[topic_info.Topic != -1])
    
    if meaningful_topics == 0:
        print("No meaningful clusters found - treating outlier cluster (-1) as the main topic")
        print(f"Found 1 outlier topic (all {len(documents)} documents classified as outliers)")
    else:
        print(f"Found {meaningful_topics} meaningful topics")
        
        # Only reduce outliers if we have meaningful topics to work with
        # Skip outlier reduction when all documents are outliers (meaningful_topics == 0)
        if len(topic_info[topic_info.Topic == -1]) > 0:
            print("Reducing outliers...")
            # Pass embeddings to use pre-computed embeddings instead of re-embedding
            new_topics = topic_model.reduce_outliers(documents, topics, embeddings=embeddings, strategy="embeddings")
            topics = new_topics  # Update topics with reduced outliers
    
    # Get reduced embeddings for visualization
    reduced_embeddings = topic_model.umap_model.embedding_
    
    return topic_model, topics, reduced_embeddings


def get_representative_docs(topic_model: BERTopic, documents: List[str], 
                          topic_id: int, n_docs: int = 25) -> List[str]:
    """
    Get representative documents for a topic using c-TF-IDF + MMR sampling.
    
    Args:
        topic_model (BERTopic): Fitted BERTopic model
        documents (List[str]): All documents
        topic_id (int): Topic ID to get representative docs for
        n_docs (int): Number of representative docs to return
        
    Returns:
        List[str]: Representative documents
    """
    # Get documents for this topic
    topic_docs_idx = [i for i, topic in enumerate(topic_model.topics_) if topic == topic_id]
    topic_docs = [documents[i] for i in topic_docs_idx]
    
    if len(topic_docs) == 0:
        return []
    
    # First get one document using c-TF-IDF (BERTopic's default method)
    # Note: For topic -1 (outliers), get_representative_docs might not work
    try:
        if topic_id == -1:
            # For outlier cluster, just use the first document
            ctfidf_doc = topic_docs[0]
        else:
            representative_docs = topic_model.get_representative_docs(topic_id)
            if representative_docs and len(representative_docs) > 0:
                ctfidf_doc = representative_docs[0]
            else:
                ctfidf_doc = topic_docs[0]  # Fallback
    except Exception as e:
        print(f"Warning: Could not get representative docs for topic {topic_id}: {e}")
        ctfidf_doc = topic_docs[0]  # Fallback
    
    # If we only need one document, return it
    if n_docs == 1:
        return [ctfidf_doc]
    
    # For additional documents, sample from the cluster
    if len(topic_docs) <= n_docs:
        # If cluster has fewer docs than requested, return all
        return topic_docs
    
    # Random sample 200 documents (or all if fewer)
    sample_size = min(200, len(topic_docs))
    sampled_docs = np.random.choice(topic_docs, size=sample_size, replace=False).tolist()
    
    # Use the c-TF-IDF doc as first, then add more from MMR-like sampling
    result_docs = [ctfidf_doc]
    remaining_docs = [doc for doc in sampled_docs if doc != ctfidf_doc]
    
    # Simple diversity sampling (pseudo-MMR without embeddings)
    for _ in range(min(n_docs - 1, len(remaining_docs))):
        # Select documents with different lengths/characteristics for diversity
        if remaining_docs:
            # Sort by length difference from already selected docs
            selected_lengths = [len(doc.split()) for doc in result_docs]
            avg_selected_length = sum(selected_lengths) / len(selected_lengths)
            
            # Find document with most different length
            best_doc = max(remaining_docs, 
                          key=lambda doc: abs(len(doc.split()) - avg_selected_length))
            result_docs.append(best_doc)
            remaining_docs.remove(best_doc)
    
    return result_docs[:n_docs]


def save_topic_results(topic_model: BERTopic, documents: List[str], topics: List, 
                      output_path: str, topic: str, platform: str) -> pd.DataFrame:
    """
    Prepare topic modeling results (but don't save yet - will be saved with LLM labels).
    
    Args:
        topic_model (BERTopic): Fitted BERTopic model
        documents (List[str]): All documents
        topics (List): Topic assignments
        output_path (str): Output directory path
        topic (str): Topic name
        platform (str): Platform name
        
    Returns:
        pd.DataFrame: Topics information dataframe
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Save BERTopic model as pickle file
    model_file = os.path.join(output_path, f"bertopic_model_{topic}_{platform}.pkl")
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(topic_model, f)
        print(f"Saved BERTopic model to {model_file}")
    except Exception as e:
        print(f"Failed to save BERTopic model: {e}")
    
    # 2. Get topic information
    topic_info = topic_model.get_topic_info()
    
    # Handle case where no meaningful clusters exist - use outlier cluster (-1)
    meaningful_topics = topic_info[topic_info.Topic != -1].copy()
    
    if len(meaningful_topics) == 0:
        # No meaningful clusters found - use the outlier cluster (-1)
        print("Using outlier cluster (-1) as the main topic")
        top_topics = topic_info[topic_info.Topic == -1].copy()
    else:
        # Normal case - filter out outlier topic (-1) and get top 10 topics
        top_topics = meaningful_topics.head(10)
    
    # 3. Prepare results dataframe
    results_data = []
    
    for _, row in top_topics.iterrows():
        topic_id = row['Topic']
        count = row['Count']
        
        # Get representative words (top 10 from c-TF-IDF)
        topic_words = topic_model.get_topic(topic_id)
        rep_words = [word for word, _ in topic_words[:10]]
        
        # Get representative documents (25 docs using c-TF-IDF + MMR sampling)
        rep_docs = get_representative_docs(topic_model, documents, topic_id, n_docs=25)
        
        results_data.append({
            'topic_id': topic_id,
            'topic_size': count,
            'representative_words': ', '.join(rep_words),
            'representative_docs': ' ||| '.join(rep_docs)  # Separate docs with |||
        })
    
    # Create results dataframe (will be saved later with LLM labels)
    results_df = pd.DataFrame(results_data)
    
    return results_df

def generate_topic_label(client, topic_docs):
    """
    Generate a topic label using OpenAI’s new Responses API (e.g., GPT-5.1).
    """
    prompt = f"""
    Based on the following documents, identify the common topic or theme they share.
    Provide a concise topic label that best describes what these documents are about.
    Only return the topic label, nothing else.

    Documents:
    [DOCUMENTS]
    """

    documents_text = "\n".join([f"- {doc}" for doc in topic_docs])
    current_prompt = prompt.replace("[DOCUMENTS]", documents_text)

    response = client.responses.create(
        model="gpt-5-nano-2025-08-07", 
        input=current_prompt,           
        max_output_tokens=None           
    )

    return response.output_text.strip()

def generate_llm_labels(topic_model: BERTopic, results_df: pd.DataFrame, 
                       output_path: str, topic: str, platform: str, api_key: str):
    """
    Generate LLM-based topic labels using OpenAI API directly and save final results.
    
    Args:
        topic_model (BERTopic): Fitted BERTopic model (not used in this step)
        results_df (pd.DataFrame): Results dataframe
        output_path (str): Output directory path
        topic (str): Topic name
        platform (str): Platform name
        api_key (str): OpenAI API key
    """
    try:
        # Create OpenAI client
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        print("Generating LLM-based topic labels...")
        print()
        
        # Create labeled results by adding LLM labels to existing results
        labeled_results = []
        
        for _, row in results_df.iterrows():
            topic_id = row['topic_id']
            
            # Extract the representative documents
            rep_docs_str = row['representative_docs']
            all_rep_docs = rep_docs_str.split(' ||| ')
            
            # Use first 15 documents for LLM labeling (MMR sampled ones)
            if len(all_rep_docs) >= 15:
                topic_docs = all_rep_docs[:15]  # First 15 docs (c-TF-IDF + MMR)
            else:
                topic_docs = all_rep_docs  # Use all available if less than 15
            
            # Generate LLM label using OpenAI API
            try:
                llm_label = generate_topic_label(client, topic_docs)
            except Exception as e:
                print(f"Failed to generate label for topic {topic_id}: {e}")
                llm_label = "Unknown"
            
            # Print topic information
            print(f"Topic {topic_id}:")
            print(f"  Keywords: {row['representative_words']}")
            print(f"  LLM Label: {llm_label}")
            print()
            
            # Add to results with LLM label
            labeled_results.append({
                'topic_id': topic_id,
                'topic_size': row['topic_size'],
                'representative_words': row['representative_words'],
                'representative_docs': row['representative_docs'],
                'llm_label': llm_label
            })
        
        # Save final results with LLM labels (only one CSV file)
        labeled_df = pd.DataFrame(labeled_results)
        csv_file = os.path.join(output_path, f"bertopic_results_{topic}_{platform}.csv")
        labeled_df.to_csv(csv_file, index=False)
        print(f"Saved final results with LLM labels to {csv_file}")
        
    except Exception as e:
        print(f"Failed to generate LLM labels: {e}")
        # Save results without LLM labels as fallback
        csv_file = os.path.join(output_path, f"bertopic_results_{topic}_{platform}.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"Saved results without LLM labels to {csv_file}")


def main(embeddings_dict_preloaded=None):
    """Main function to run the BERTopic modeling pipeline."""
    parser = argparse.ArgumentParser(description='Run BERTopic modeling pipeline')
    parser.add_argument('--input_parquet', required=True, 
                       help='Path to input parquet file with document data')
    parser.add_argument('--embedding_folder', required=True,
                       help='Path to folder containing embedding files')
    parser.add_argument('--output_path', required=True,
                       help='Path to output directory for results')
    parser.add_argument('--topic', required=True,
                       help='Topic value to filter documents')
    parser.add_argument('--platform', required=True,
                       help='Platform value to filter documents')
    parser.add_argument('--min_cluster_size', type=int, default=15,
                       help='Minimum cluster size for HDBSCAN (default: 15)')
    
    args = parser.parse_args()
    
    try:
        print("="*60)
        print("BERTOPIC MODELING PIPELINE")
        print("="*60)
        
        # Step 1: Load embeddings (skip if preloaded)
        if embeddings_dict_preloaded is not None:
            print("\nStep 1: Using pre-loaded embeddings...")
            embeddings_dict = embeddings_dict_preloaded
            print(f"Total embeddings available: {len(embeddings_dict)}")
        else:
            print("\nStep 1: Loading embeddings...")
            embeddings_dict = load_all_embeddings(args.embedding_folder)
        
        if not embeddings_dict:
            raise ValueError("No embeddings loaded. Check embedding folder path.")
        
        # Step 2: Load and filter documents
        print("\nStep 2: Loading and filtering documents...")
        df = pd.read_parquet(args.input_parquet)
        print(f"Loaded {len(df)} documents from parquet file")
        
        # Verify required columns
        required_cols = ['platform', 'topic', 'embed_id', 'post_clean']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        documents, doc_ids, doc_embeddings = filter_and_prepare_data(
            df, args.topic, args.platform, embeddings_dict, args.min_cluster_size
        )
        
        # Step 3: Run BERTopic modeling
        print("\nStep 3: Running BERTopic modeling...")
        topic_model, topics, reduced_embeddings = create_bertopic_model(
            documents, doc_embeddings, args.min_cluster_size
        )
        
        # Step 4: Save results
        print("\nStep 4: Saving results...")
        results_df = save_topic_results(
            topic_model, documents, topics, args.output_path, args.topic, args.platform
        )
        
        # Step 5: Generate LLM labels
        openai_api_key = os.environ.get("OPENAI_API_KEY_calvin1")
        
        if openai_api_key:
            print("\nStep 5: Generating LLM topic labels...")
            generate_llm_labels(
                topic_model, results_df, args.output_path, 
                args.topic, args.platform, openai_api_key
            )
        else:
            print("\nStep 5: Skipped LLM labeling (no API key provided)")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {args.output_path}")
        print(f"Topics found: {len(results_df)}")
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()