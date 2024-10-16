# Topological Data Analysis: Human-Written vs. AI-Generated Essays
This is our ✨ repo ✨ for the SF2956 TDA project @ KTH Stockholm

## Structure
* ``data``: the folder with the data processing steps and the embedding retrieval script
* ``sandbox``: the folder with the random notebooks/scripts to test ideas

## Data

The dataset of human-written and AI-generated essays is taken from *Shijaku, Rexhep & Canhasi, Ercan. (2023) ChatGPT Generated Text Detection*: [GitHub][1]. The processed dataset can be found [here][2]. Additionally, the file ``sentence_embeddings.npy`` with the sentence embeddings is located there. The order of embeddings exactly corresponds to the dataset's ``embedding_id`` column. We used the open-source [NV-Embed-v2][3] as the embedding model of choice.


[1]: https://github.com/rexshijaku/chatgpt-generated-text-detection-corpus?tab=readme-ov-file
[2]: 
[3]: https://huggingface.co/nvidia/NV-Embed-v2
[4]
