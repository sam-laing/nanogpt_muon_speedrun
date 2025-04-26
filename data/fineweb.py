"""
FineWeb dataset preprocessing for SRS pretraining
Improved version with better memory handling and error checking
"""

import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def write_datafile(filename, tokens):
    """ 
    Saves token data as a .bin file with header and uint16 tokens.
    Args:
        filename: Output file path
        tokens: List or numpy array of tokens to save
    """
    try:
        # Validate input
        if not isinstance(tokens, (list, np.ndarray)):
            raise ValueError("Tokens must be list or numpy array")
            
        if len(tokens) >= 2**31:
            raise ValueError("Token count exceeds maximum (2^31)")
            
        # Convert to numpy array if needed
        if not isinstance(tokens, np.ndarray) or tokens.dtype != np.uint16:
            tokens_np = np.array(tokens, dtype=np.uint16)
        else:
            tokens_np = tokens
            
        # Validate token values
        if np.any(tokens_np >= 2**16):
            raise ValueError("Token values exceed uint16 maximum")
            
        # Create header
        header = np.zeros(256, dtype=np.int32)
        header[0] = 20240520  # magic number
        header[1] = 1         # version
        header[2] = len(tokens_np)  # token count
        
        # Write to file
        print(f"Writing {len(tokens_np):,} tokens to {filename}")
        with open(filename, "wb") as f:
            f.write(header.tobytes())
            f.write(tokens_np.tobytes())
            
    except Exception as e:
        print(f"Error writing {filename}: {str(e)}")
        raise

def tokenize_document(doc, enc, eot_token):
    """
    Tokenizes a single document with EOT token prefix.
    Args:
        doc: Document dictionary with 'text' field
        enc: Tokenizer instance
        eot_token: End-of-text token ID
    Returns:
        numpy array of uint16 tokens
    """
    try:
        tokens = [eot_token]  # Start with EOT token
        tokens.extend(enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens, dtype=np.uint16)
        return tokens_np
    except Exception as e:
        print(f"Error tokenizing document: {str(e)}")
        return np.array([eot_token], dtype=np.uint16)  # Return minimal valid output

def process_shard(shard_tokens, shard_index, output_dir, is_first_shard=False):
    """Processes and writes a single shard of tokens."""
    split = "val" if is_first_shard else "train"
    filename = os.path.join(output_dir, f"fineweb_{split}_{shard_index:06d}.bin")
    write_datafile(filename, shard_tokens)

def main():
    parser = argparse.ArgumentParser(description="FineWeb dataset preprocessing")
    parser.add_argument("-v", "--version", type=str, default="10B", 
                       choices=["10B", "100B"], help="Dataset version")
    parser.add_argument("-s", "--shard_size", type=int, default=10**8, 
                       help="Tokens per shard (default: 100M)")
    parser.add_argument("-o", "--output_dir", type=str, 
                       default="/fast/slaing/data/lm/fineweb",
                       help="Base output directory")
    parser.add_argument("-j", "--workers", type=int, 
                       default=max(1, os.cpu_count() - 2),
                       help="Number of worker processes")
    args = parser.parse_args()

    # Validate and setup paths
    local_dir = f"fineweb{args.version}"
    remote_name = f"sample-{args.version}T"
    output_dir = os.path.join(args.output_dir, local_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading FineWeb-{args.version} dataset...")
    dataset = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc._special_tokens['<|endoftext|>']

    # Process documents
    shard_index = 0
    current_shard = []
    current_count = 0
    is_first_shard = True

    with mp.Pool(args.workers) as pool, \
         tqdm(desc="Processing documents", unit="doc") as pbar:

        # Create tokenization partial function
        tokenize_fn = lambda doc: tokenize_document(doc, enc, eot_token)

        for tokens in pool.imap(tokenize_fn, dataset, chunksize=16):
            tokens_len = len(tokens)
            
            # Check if we need to start a new shard
            if current_count + tokens_len >= args.shard_size:
                # Write current shard
                process_shard(current_shard, shard_index, output_dir, is_first_shard)
                
                # Start new shard
                shard_index += 1
                is_first_shard = False
                remaining_space = args.shard_size - current_count
                
                # Split current document if needed
                if remaining_space > 0:
                    current_shard = list(tokens[:remaining_space])
                    current_count = len(current_shard)
                    tokens = tokens[remaining_space:]
                else:
                    current_shard = []
                    current_count = 0
                    
            # Add tokens to current shard
            current_shard.extend(tokens)
            current_count += len(tokens)
            pbar.update(1)

    # Write any remaining tokens in the final shard
    if current_count > 0:
        process_shard(current_shard, shard_index, output_dir, is_first_shard)

    print(f"Processing complete. Saved {shard_index + 1} shards to {output_dir}")

if __name__ == "__main__":
    main()