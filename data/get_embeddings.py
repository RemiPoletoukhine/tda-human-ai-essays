import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def get_embeddings(sentences: list[str], model: str = 'nvidia/NV-Embed-v2', instruction: str = "", max_length: int = 32768, batch_size: int = 20, num_workers: int = 24) -> None:
    """
    Get the embeddings of the text using the model from HF's transformers library.
    
    Args:
        sentences: the sentences to embed.
        model: the model from HF's transformers library.
        instruction: the instruction. Not needed for retrieval passages.
        max_length: the maximum length of the input text.
        batch_size: the size of the mini-batches.
        num_workers: the number of workers to use for the DataLoader.

    Returns:
        None
    """
    # load model with tokenizer
    model = AutoModel.from_pretrained(model, trust_remote_code=True)
    # get the embeddings with DataLoader (spliting the datasets into multiple mini-batches)
    sentence_embeddings = model._do_encode(sentences, batch_size=batch_size, instruction=instruction, max_length=max_length, num_workers=num_workers, return_numpy=True)
    # save the embeddings
    np.save('sentence_embeddings.npy', sentence_embeddings)
    print("Embeddings saved successfully.")
    
    return 0


if __name__ == "__main__":
    # load the dataframe with the processed dataset
    essay_dataset = pd.read_csv('essay_dataset.csv')
    # get the sentences
    sentences = essay_dataset['sentence'].tolist()
    # get the embeddings
    _ = get_embeddings(sentences)
