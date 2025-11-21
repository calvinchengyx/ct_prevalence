# topic - pre-embed data before BERTopic modeling for more efficient processing
# notes
# - input: cleaned data with 'post_clean' column
# - output: embeddings npy file where embeddings and text data are linked by 'embed_id' column (which is the string of the row index)
# argument parsing: input_path, output_path, and embed_model_name
# ...existing code...
import os
import datetime
import gc
import argparse
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add OpenAI import
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

tqdm.pandas()

# set the gpu device to the 3rd one
torch.cuda.set_device(2)
# Clear any existing CUDA memory first
torch.cuda.empty_cache()
gc.collect()

def prepare_docs_and_ids(df: pd.DataFrame, text_col: str = "post_clean", id_col: str = "embed_id"):
    """Prepare documents and IDs for embedding, filtering out invalid entries."""
    if text_col not in df.columns:
        raise ValueError(f"Input dataframe missing text column: {text_col}")
    
    if id_col not in df.columns:
        # Create embed IDs if not present
        df = df.reset_index(drop=True)
        df[id_col] = df.index.astype(str)

    # Convert to lists and validate strings
    ids = df[id_col].astype(str).tolist()
    docs = df[text_col].tolist()
    
    clean_docs = []
    clean_ids = []
    
    for i, doc in enumerate(docs):
        if isinstance(doc, str) and len(doc.strip()) > 0:
            clean_docs.append(doc.strip())
            clean_ids.append(ids[i])
    
    print(f"Filtered {len(clean_docs)} valid documents from {len(docs)} total")
    return clean_docs, clean_ids

def generate_and_save_embeddings_openai(docs, ids, output_path, openai_model="text-embedding-3-small", 
                                       chunk_size=1000):
    """Generate embeddings using OpenAI API and save them in chunks."""
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please pip install openai")
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY_calvin1")
    if not api_key:
        raise RuntimeError("OpenAI API key not found in environment variable 'OPENAI_API_KEY_calvin1")
    
    os.makedirs(output_path, exist_ok=True)
    client = OpenAI(api_key=api_key)
    
    # Split into chunks (smaller for OpenAI API)
    docs_chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
    ids_chunks = [ids[x:x+chunk_size] for x in range(0, len(ids), chunk_size)]
    
    for i in range(len(docs_chunks)):
        out_file = f"{output_path}/embeddings_{i+1}.npy"
        if os.path.isfile(out_file):
            print(f"Chunk {i+1} already exists, skipping...")
            continue
            
        print(f"Processing chunk {i+1} of {len(docs_chunks)} (OpenAI)")
        print(f"Starting at {datetime.datetime.now()}")
        
        try:
            # Call OpenAI embeddings API
            response = client.embeddings.create(
                model=openai_model,
                input=docs_chunks[i]
            )
            embeddings = [r.embedding for r in response.data]
            embeddings = np.array(embeddings)
            embeddings_dict = dict(zip(ids_chunks[i], embeddings))
            
            # Save chunk to file
            np.save(out_file, embeddings_dict)
            print(f"Saved {len(embeddings_dict)} embeddings to {out_file}")
            
        except Exception as e:
            print(f"Failed to generate embeddings for chunk {i+1}: {e}")
        finally:
            # Cleanup
            if 'embeddings' in locals():
                del embeddings
            if 'embeddings_dict' in locals():
                del embeddings_dict
            gc.collect()
    
    print("OpenAI embedding generation completed!")

def generate_and_save_embeddings_sentence_transformer(docs, ids, output_path, model_name="all-mpnet-base-v2", 
                                                     chunk_size=10000, device="cuda:0"):
    """Generate embeddings for documents and save them in chunks using sentence-transformers."""
    os.makedirs(output_path, exist_ok=True)
    
    # Set GPU device
    if torch.cuda.is_available():
        try:
            device_id = int(device.split(':')[1]) if ':' in device else 0
            torch.cuda.set_device(device_id)
            print(f"Using device: {torch.cuda.get_device_properties(device_id)}")
        except:
            device = "cpu"
            print("CUDA device setup failed, using CPU")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    # Load embedding model
    print(f"Loading {model_name} model...")
    model = SentenceTransformer(f'sentence-transformers/{model_name}', device=device)
    print(f"Model device: {next(model.parameters()).device}")
    
    # Split into chunks
    docs_chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
    ids_chunks = [ids[x:x+chunk_size] for x in range(0, len(ids), chunk_size)]
    
    for i in range(len(docs_chunks)):
        out_file = f"{output_path}/embeddings_{i+1}.npy"
        if os.path.isfile(out_file):
            print(f"Chunk {i+1} already exists, skipping...")
            continue
            
        print(f"Processing chunk {i+1} of {len(docs_chunks)} (Sentence Transformer)")
        print(f"Starting at {datetime.datetime.now()}")
        
        # Generate embeddings for this chunk
        embeddings = model.encode(docs_chunks[i], show_progress_bar=True, 
                                batch_size=32, device=device)
        embeddings_dict = dict(zip(ids_chunks[i], embeddings))
        
        # Save chunk to file
        np.save(out_file, embeddings_dict)
        print(f"Saved {len(embeddings_dict)} embeddings to {out_file}")
        
        # Clear GPU memory
        del embeddings
        del embeddings_dict
        torch.cuda.empty_cache()
        gc.collect()
        
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            print(f'Memory after cleanup: {memory_gb:.1f} GB')
    
    print("Sentence Transformer embedding generation completed!")

def run(model_name: str, input_path: str, output_path: str, chunk_size: int = 10000, 
        device: str = "cuda:0"):
    """Main function to run embedding generation."""
    print(f"Reading input file: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Prepare documents and IDs
    docs, ids = prepare_docs_and_ids(df)
    print(f"Prepared {len(docs)} documents for embedding")
    
    # Choose embedding method based on model name
    if model_name.lower() == "openai":
        # Use default OpenAI model or allow specification
        generate_and_save_embeddings_openai(docs, ids, output_path, 
                                           openai_model="text-embedding-3-small", 
                                           chunk_size=min(chunk_size, 1000))  # OpenAI has smaller limits
    else:
        # Use sentence-transformers
        generate_and_save_embeddings_sentence_transformer(docs, ids, output_path, model_name, 
                                                         chunk_size, device)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Embed textual data and save embeddings.")
    p.add_argument("--model", required=True, help="Model name (e.g., all-mpnet-base-v2, openai)")
    p.add_argument("--input", "-i", required=True, help="Input cleaned parquet file path")
    p.add_argument("--output", "-o", required=True, help="Output embeddings folder path")
    p.add_argument("--chunk-size", type=int, default=10000, help="Number of documents per chunk (auto-adjusted for OpenAI)")
    p.add_argument("--device", default="cuda:0", help="Device to use (e.g., cuda:0, cuda:1, cpu)")
    return p.parse_args()

# Only run this python script directly (not imported as a module)
if __name__ == "__main__":
    args = parse_args()
    run(args.model, args.input, args.output, args.chunk_size, args.device)