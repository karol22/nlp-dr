# nlp-dr
Code snippets for NLP project


then install torch manually, such as 
conda install pytorch torchvision -c pytorch

# NLP-DR: Text summarization

Sample script to summarize text using the Hugging Face `transformers` library.
The script takes text fragments, generates summaries and generate plots.

## Project Structure

- **summarization_example.py**: The main script that performs text summarization and plots.

## Setup Instructions

### Prerequisites

- **Python 3.9** or later.
- **Miniconda or Anaconda** installed on your system.

### Setup the Environment

1. Run `conda env create -f environment.yaml`
2. Activate environment using `conda activate haag-nlp`
3. Manually install pytorch, use reference from `https://pytorch.org/get-started/locally/`.
    For example: `conda install pytorch torchvision -c pytorch`

### Run the code

Use `python summarization_example.py`.
