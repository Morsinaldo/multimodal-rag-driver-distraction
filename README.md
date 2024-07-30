## Multimodal RAG for Classifying Driver Distraction

[![Python](https://img.shields.io/badge/Python-3.9-black.svg)](https://www.python.org/)

This repository contains the implementation of a Multimodal Retrieval Augmented Generation (RAG) to classify driver distraction.

### Overview

The project begins by utilizing the [Driver Distraction dataset](https://universe.roboflow.com/new-workspace-vrhvx/distracted-driver-detection), which comprises both textual and image data. Upon downloading the dataset, [OpenClip embeddings](https://python.langchain.com/docs/integrations/text_embedding/open_clip/) are utilized to vectorize the data, which is then stored in the [Chroma database](https://www.trychroma.com/), forming the knowledge base. Subsequently, users can input a multimodal prompt consisting of both an image and text to the Visual Language Model (VLM), which consults the knowledge base to generate a response based on the most relevant findings. In this case, the VLM used is GPT4-Turbo.

### How to Execute
1. Clone the repository:
    ```bash
    git clone https://github.com/Morsinaldo/multimodal-rag-driver-distraction.git
    ```
2. Change the directory:
    ```bash
    cd multimodal-rag-driver-distraction
    ```
3. Create a virtual environment:
    ```bash
    conda create --name rag python==3.9
    ```
4. Activate the virtual environment:
    ```bash
    conda activate rag
    ```
5. Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
6. Download the dataset from [Roboflow](https://universe.roboflow.com/new-workspace-vrhvx/distracted-driver-detection)
7. In the [notebook](./notebook.ipynb) file, put your OpenAI's API key and adjust the path to the images directory.
    ```
    # environment variables
    OPENAI_API_KEY = 'YOUR_API_KEY'
    MODEL_NAME = 'gpt-4.0-turbo'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMAGES_DIR = 'images/train'
    ```
With this, you will be able to execute the notebook.

### Results and Discussions

