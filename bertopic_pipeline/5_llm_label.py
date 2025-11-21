#!/usr/bin/env python3
"""
Regenerate Topic Labels with GPT-5.1 Reasoning Model

This script reads existing BERTopic results and regenerates topic labels
using OpenAI's GPT-5.1 reasoning model for better quality labels.

Usage:
    python regenerate_labels_gpt5.py --input_csv <path> --output_csv <path>
"""

import argparse
import os
import sys
import pandas as pd
from openai import OpenAI


def generate_topic_label_gpt5(client, representative_words, representative_docs):
    """
    Generate a topic label using OpenAI's GPT-5.1 reasoning model.
    
    Args:
        client: OpenAI client
        representative_words (str): Comma-separated representative words
        representative_docs (str): Representative documents separated by |||
        
    Returns:
        str: Generated topic label
    """
    # Parse documents
    docs_list = representative_docs.split(' ||| ')
    
    # Use first 15 documents for labeling
    docs_to_use = docs_list[:15] if len(docs_list) >= 15 else docs_list
    
    # Create prompt
    prompt = f"""You are a topic modeling assistant that helps summarize conspiracy topics given the representative words and documents of clusters.
                Based on the following information, briefly describe the topic of this cluster. Provide a concise label (maximum 10 words) that captures the main conspiracy theory or theme.
                Representative Keywords: {representative_words}
                Representative Documents:{chr(10).join([f"- {doc}" for doc in docs_to_use])}
                Provide only the topic label, nothing else."""

    try:
        # Use GPT-5.1 reasoning model with Responses API
        response = client.responses.create(
            model="gpt-5.1-2025-11-13",
            input=prompt,
            max_output_tokens=50  # Limit output length for concise labels
        )
        
        return response.output_text.strip()
    
    except Exception as e:
        print(f"Error generating label: {e}")
        return "Error: Could not generate label"


def regenerate_labels(df, api_key):
    """
    Regenerate labels for all topics in the dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with topic results
        api_key (str): OpenAI API key
        
    Returns:
        pd.DataFrame: Dataframe with new gpt5.1 column
    """
    # Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    print(f"Regenerating labels for {len(df)} topics using GPT-5.1...")
    print()
    
    # Create list to store new labels
    new_labels = []
    
    for idx, row in df.iterrows():
        topic_id = row.get('topic_id', idx)
        representative_words = row['representative_words']
        representative_docs = row['representative_docs']
        
        print(f"Processing topic {topic_id}...")
        
        # Generate new label
        new_label = generate_topic_label_gpt5(
            client, 
            representative_words, 
            representative_docs
        )
        
        new_labels.append(new_label)
        
        print(f"  Keywords: {representative_words[:100]}...")
        print(f"  New GPT-5.1 Label: {new_label}")
        print()
    
    # Add new column to dataframe
    df['gpt5.1'] = new_labels
    
    return df


def main():
    """Main function to regenerate topic labels."""
    parser = argparse.ArgumentParser(
        description='Regenerate topic labels using GPT-5.1 reasoning model'
    )
    parser.add_argument('--input_csv', required=True,
                       help='Path to input CSV file with topic results')
    parser.add_argument('--output_csv', required=True,
                       help='Path to save output CSV file with new labels')
    
    args = parser.parse_args()
    
    try:
        print("="*60)
        print("REGENERATE TOPIC LABELS WITH GPT-5.1")
        print("="*60)
        
        # Load API key from environment
        api_key = os.environ.get("OPENAI_API_KEY_calvin1")
        if not api_key:
            raise ValueError(
                "API key not found. Please set OPENAI_API_KEY_calvin1 environment variable."
            )
        
        print("\nStep 1: Loading existing results...")
        df_results = pd.read_csv(args.input_csv)
        print(f"Loaded {len(df_results)} topics from {args.input_csv}")
        
        # Verify required columns
        required_cols = ['representative_words', 'representative_docs']
        missing_cols = [col for col in required_cols if col not in df_results.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"\nColumns in dataframe: {df_results.columns.tolist()}")
        
        print("\nStep 2: Regenerating labels with GPT-5.1...")
        df_results = regenerate_labels(df_results, api_key)
        
        print("\nStep 3: Saving results...")
        df_results.to_csv(args.output_csv, index=False)
        print(f"Saved results to {args.output_csv}")
        
        print("\n" + "="*60)
        print("LABEL REGENERATION COMPLETED!")
        print("="*60)
        print(f"\nNew column 'gpt5.1' added with {len(df_results)} labels")
        
        # Show sample of results
        print("\nSample results:")
        print(df_results[['topic_id', 'llm_label', 'gpt5.1']].head() if 'llm_label' in df_results.columns 
              else df_results[['gpt5.1']].head())
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()