# # Fast inference with Infinity (mixedbread-ai/mxbai-rerank-large-v1)

import os
import subprocess
import secrets

from modal import Image, Secret, Stub, enter, gpu, method, web_server

MODEL_DIR = "/model"
BASE_MODEL = "mixedbread-ai/mxbai-rerank-large-v1"

# ## Define a container image


# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# Tip: avoid using global variables in this function. Changes to code outside this function will not be detected and the download step will not re-run.
def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()


# ### Image definition
# We'll start from a recommended Docker Hub image and install `vLLM`.
# Then we'll use `run_function` to run the function defined above to ensure the weights of
# the model are saved within the container image.
image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        "wheel==0.43.0",
        "huggingface_hub==0.23.0",
        "hf-transfer==0.1.6",
        "torch==2.2.1",
        "poetry==1.8.2",
        "transformers==4.40.1",
        "sentence-transformers==2.6.1",
    )
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/monotykamary/infinity.git",
        "cd infinity/libs/infinity_emb && git checkout c8121b9e19fcd7658aa87aea2457979b07c9fd25 && poetry build && pip install 'dist/infinity_emb-0.0.32-py3-none-any.whl[all]'",
    )
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secrets=[Secret.from_name("huggingface")],
        timeout=60 * 20,
    )
)

stub = Stub("infinity-mxbai-rerank-large-v1", image=image)
GPU_CONFIG = gpu.T4(count=1)


# Run a web server on port 7997 and expose the Infinity embedding server
@stub.function(
    allow_concurrent_inputs=100,
    container_idle_timeout=60,
    gpu=GPU_CONFIG,
    secrets=[
        Secret.from_name("huggingface"),
        Secret.from_dotenv(),
    ],
)
@web_server(7997, startup_timeout=300)
def infinity_embeddings_server():
    cmd = f"infinity_emb --device cuda --engine torch --model-name-or-path {BASE_MODEL}"
    subprocess.Popen(cmd, shell=True)
