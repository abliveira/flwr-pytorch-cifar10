---
tags: [quickstart, tabular, federated]
dataset: [UCI HAR]
framework: [torch]
---

# Federated Human Activity Recognition with PyTorch and Flower

This Flower example trains a PyTorch classifier on the local [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). It uses the dataset's 561-feature vectors to classify the six activities. Each simulated client is assigned the official training rows for exactly one participant. The server evaluates against the official test split.

## Set up the project

Flower's app bundle excludes the dataset's `.txt` files. The absolute path to the existing dataset directory is configured as `dataset-dir` in `pyproject.toml`; the server and each client read the files directly from that location without copying or downloading them.

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` and the project:

```bash
pip install -e .
```

## Run the project

The project can run in Flower's _simulation_ and _deployment_ modes. Simulation is the simplest way to get started:

### Run with the Simulation Engine

```powershell
flwr run . --federation-config="num-supernodes=21" --stream
```

The 21 clients map by sorted `subject_train.txt` ID to the 21 participants in the official training split. Configure the simulation with exactly 21 clients; the app rejects a different client count rather than mixing participants or silently omitting data.

Run settings such as the number of rounds and learning rate can be overridden:

```bash
flwr run . --federation-config="num-supernodes=21" --run-config "num-server-rounds=5 learning-rate=0.05" --stream
```

For more information, see the [Flower PyTorch quickstart tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html).
