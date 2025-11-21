#!/usr/bin/env python3
"""
Example usage script for BERTopic modeling pipeline

This script demonstrates how to use the bertopic_modeling.py script
with different configurations and datasets.
"""

import os
import sys
from pathlib import Path

# Import the bertopic_modeling module to use its functions directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bertopic_modeling import main as bertopic_main, load_all_embeddings

def run_bertopic_analysis(embeddings_dict, input_parquet, embedding_folder, output_path, 
                         topic, platform, min_cluster_size=10):
    """
    Run BERTopic analysis with pre-loaded embeddings.
    
    Args:
        embeddings_dict (dict): Pre-loaded embeddings dictionary
        input_parquet (str): Path to input parquet file
        embedding_folder (str): Path to embeddings folder
        output_path (str): Output directory path
        topic (str): Topic to analyze
        platform (str): Platform to analyze
        min_cluster_size (int): Minimum cluster size
    """
    # Set up sys.argv to simulate command line arguments
    original_argv = sys.argv
    sys.argv = [
        'bertopic_modeling.py',
        '--input_parquet', input_parquet,
        '--embedding_folder', embedding_folder,
        '--output_path', output_path,
        '--topic', topic,
        '--platform', platform,
        '--min_cluster_size', str(min_cluster_size)
    ]
    
    try:
        # Call the main function with pre-loaded embeddings
        bertopic_main(embeddings_dict_preloaded=embeddings_dict)
        print(f"SUCCESS: Analysis completed for {topic} on {platform}")
        return True
    except Exception as e:
        print(f"ERROR: Analysis failed for {topic} on {platform}")
        print(f"Error: {e}")
        return False
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def main():
    """
    Example usage of the BERTopic modeling pipeline.
    """
    # Configuration
    base_path = "/VData/scro4316/ct_prevalence"
    
    # Input files
    input_parquet = f"{base_path}/calvin_posts_cleaned.parquet"  # Update with actual path
    embedding_folder = f"{base_path}/embeddings_openai"  # Update with actual path
    output_base = f"{base_path}/bertopic_results"
    
    # Create output directory
    Path(output_base).mkdir(exist_ok=True)
    
    # Topics and platforms to analyze
    topics = [ 'KEYWORDS_MOON','KEYWORDS_COVID19', 'KEYWORDS_NWO', 'KEYWORDS_9_11', 'KEYWORDS_ALIEN']
    platforms = ['fediverse','truthsocial', 'gab', 'X', '4chan','bluesky', 'gettr']  # Update with actual platform names
    
    # Set OpenAI API key as environment variable (if available)
    openai_api_key = os.getenv('OPENAI_API_KEY_calvin1')  # or os.getenv('OPENAI_API_KEY_calvin1') if you prefer
    if openai_api_key:
        os.environ['OPENAI_API_KEY'] = openai_api_key
        print(f"OpenAI API key loaded from environment variable")
    else:
        print(f"No OpenAI API key found - LLM labeling will be skipped")
    
    # *** LOAD EMBEDDINGS ONCE AT THE BEGINNING ***
    print("="*60)
    print("LOADING EMBEDDINGS (ONE TIME ONLY)")
    print("="*60)
    embeddings_dict = load_all_embeddings(embedding_folder)
    if not embeddings_dict:
        print("ERROR: Failed to load embeddings!")
        return
    print(f"Loaded {len(embeddings_dict)} embeddings successfully")
    print("="*60)
    print()
    
    # Run analysis for each topic-platform combination
    successful_runs = []
    failed_runs = []
    
    for topic in topics:
        for platform in platforms:
            print(f"\n{'='*50}")
            print(f"Analyzing: {topic} on {platform}")
            print(f"{'='*50}")
            
            # Create output directory for this combination
            output_path = os.path.join(output_base, f"{topic}_{platform}")
            Path(output_path).mkdir(exist_ok=True)
            
            # Run the analysis with pre-loaded embeddings
            success = run_bertopic_analysis(
                embeddings_dict=embeddings_dict,  # Pass pre-loaded embeddings
                input_parquet=input_parquet,
                embedding_folder=embedding_folder,
                output_path=output_path,
                topic=topic,
                platform=platform,
                min_cluster_size=15  # Adjust as needed
            )
            
            if success:
                successful_runs.append(f"{topic}_{platform}")
            else:
                failed_runs.append(f"{topic}_{platform}")
    
    # Summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Successful runs: {len(successful_runs)}")
    for run in successful_runs:
        print(f"  ✓ {run}")
    
    print(f"\nFailed runs: {len(failed_runs)}")
    for run in failed_runs:
        print(f"  ✗ {run}")
    
    print(f"\nResults saved to: {output_base}")

if __name__ == "__main__":
    main()