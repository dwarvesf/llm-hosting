## Overview
This repository is designed for deploying and managing server processes that handle embeddings using the Infinity Embedding model or Large Language Models with an OpenAI compatible vLLM server using Modal.

## Key Components
1. **vllm_deepseek_coder_33b.py, vllm_llama3-8b.py, vllm_seallm_7b_v2_5.py, vllm_sqlcoder_7b_2.py**
   - These scripts contain the function `openai_compatible_server()` which initiates an OpenAI compatible vLLM server by running a command that instantiates an OpenAI compatible FastAPI server..
   - The `BASE_MODEL` variable appears to define the model path for the embedding tool, which is not shown but can be inferred from the context.

2. **infinity_mxbai_embed_large_v1.py, infinity_mxbai_rerank_large_v1.py, infinity_snowflake_arctic_embed_l_335m.py**
   - These scripts contain the function `infinity_embeddings_server()` which initiates the Infinity Embed server by running a command that utilizes the Infinity embedding tool with specified options (like CUDA device and Torch engine).
   - The `BASE_MODEL` variable appears to define the model path for the embedding tool, which is not shown but can be inferred from the context.

3. **devbox.json**
   - This configuration file specifies the programming environment for the repository, including versions of Python, Pip, and Node.js.
   - It also defines shell initialization hooks like activating a Python virtual environment and installing necessary Python packages, among other administration scripts.

4. **.env.example**
   - This file template shows environment variables that are likely necessary for the project to run (e.g., API keys for Infinity API and VLLM API).
   
## Prerequisites
Before diving into the project setup, make sure to:
- [Have Devbox installed](https://www.jetify.com/devbox/docs/installing_devbox/), as it manages the development and operation environment for this project.
- Set up necessary API keys by copying `.env.example` to `.env` and filling in the required values for `INFINITY_API_KEY` and `VLLM_API_KEY`.

## Environment Setup
1. **Initializing Development Environment with Devbox:**
   - Enter the Devbox shell environment by running:
     ```bash
     devbox shell
     ```
   - This action will set up the environment according to the `init_hook` specified in `devbox.json`, which activates the Python virtual environment and installs the required packages.

## Deployment
The scripts available in the repository can be deployed using the [Modal](https://modal.com/docs/examples/hello_world) tool. Deploy a script by running the corresponding command:
```bash
modal deploy infinity_mxbai_embed_large_v1.py
modal deploy infinity_mxbai_rerank_large_v1.py
modal deploy infinity_snowflake_arctic_embed_l_335m.py

modal deploy vllm_deepseek_coder_33b.py
modal deploy vllm_llama3-8b.py
modal deploy vllm_seallm_7b_v2_5.py
modal deploy vllm_sqlcoder_7b_2.py
```
Each command will deploy the respective script, launching the Infinity embeddings server or an OpenAI compatible vLLM server configured per the script's specifications.
