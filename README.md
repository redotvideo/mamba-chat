# Mamba-Chat 🐍

Mamba-Chat is the first chat language model based on a state-space model architecture, not a transformer.

The model is based on Albert Gu's and Tri Dao's work *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* as well as their [model implementation](https://github.com/state-spaces/mamba). This repository provides training / fine-tuning code for the model based on some modifications of the Huggingface Trainer class.


## Run Mamba-Chat

We provide code that lets you run inference on mamba-chat as well as our fine-tuning code. To get started, clone this repository and install its dependencies:

```
git clone https://github.com/havenhq/mamba-chat.git

cd mamba-chat
pip install -r requirements.txt
```

You can chat with mamba by 
