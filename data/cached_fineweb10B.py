import os
import sys
from huggingface_hub import hf_hub_download, login
from hf_token import TOKEN

""" 
# Download the GPT-2 tokens of Fineweb10B from huggingface
def get(fname):
    # Use the specified directory instead of a local one
    local_dir = "/fast/slaing/data/lm/fineweb/fineweb10B"
    
    # Ensure the directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Download if file doesn't exist
    if not os.path.exists(os.path.join(local_dir, fname)):
        print(f"Downloading {fname} to {local_dir}")
        hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                        repo_type="dataset", local_dir=local_dir)
    else:
        print(f"File already exists: {os.path.join(local_dir, fname)}")

if __name__=="__main__":
    login(TOKEN)
    # Download validation file
    get("fineweb_val_%06d.bin" % 0)

    # Download training files
    num_chunks = 103  # full fineweb10B. Each chunk is 100M tokens
    if len(sys.argv) >= 2:  # we can pass an argument to download less
        num_chunks = int(sys.argv[1])

    print(f"Downloading {num_chunks} training chunks...")
    for i in range(1, num_chunks+1):
        get("fineweb_train_%06d.bin" % i)

    print(f"Download complete. Files saved to /fast/slaing/data/lm/fineweb/fineweb10B")
""" 

import os
import sys
import time
from tqdm import tqdm
from huggingface_hub import hf_hub_download, login
from hf_token import TOKEN

os.environ["HF_HOME"] = "/path/to/your/custom/cache"  # Change this path
print(f"Using HuggingFace cache at: {os.environ['HF_HOME']}")

# Download with progress tracking
def get(fname, timeout=600):  # 10-minute timeout
    local_dir = "/fast/slaing/data/lm/fineweb/fineweb10B"
    os.makedirs(local_dir, exist_ok=True)
    full_path = os.path.join(local_dir, fname)
    
    if os.path.exists(full_path):
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"File already exists: {full_path} ({size_mb:.1f} MB)")
        return True
    
    print(f"Downloading {fname} to {local_dir}")
    try:
        # Start download
        start_time = time.time()
        file_downloaded = False
        
        # Start the download
        result = hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2", 
            filename=fname,
            repo_type="dataset", 
            local_dir=local_dir,
            local_files_only=False
        )
        
        # Check if file now exists
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"✓ Downloaded {fname} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"✗ Download completed but file not found at {full_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading {fname}: {str(e)}")
        return False

if __name__ == "__main__":
    # Log in
    try:
        login(TOKEN)
        print("✓ Successfully logged in to HuggingFace")
    except Exception as e:
        print(f"✗ Login failed: {str(e)}")
        print("Continuing without authentication (may hit rate limits)")
    
    # Download validation file
    if not get("fineweb_val_000000.bin"):
        print("Failed to download validation file. Exiting.")
        sys.exit(1)

    # Download training files
    num_chunks = 103
    if len(sys.argv) >= 2:
        num_chunks = int(sys.argv[1])

    print(f"Downloading {num_chunks} training chunks...")
    success_count = 0
    for i in range(1, num_chunks+1):
        print(f"[{i}/{num_chunks}] ", end="", flush=True)
        if get(f"fineweb_train_{i:06d}.bin"):
            success_count += 1
        time.sleep(1)  # Small pause between downloads
    
    print(f"\nDownload complete. Successfully downloaded {success_count}/{num_chunks} chunks.")

