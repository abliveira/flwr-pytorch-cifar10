---
tags: [quickstart, tabular, federated]
dataset: [UCI HAR]
framework: [torch]
---

# Federated Human Activity Recognition with PyTorch and Flower

This Flower example trains a PyTorch classifier on the local [UCI Human Activity Recognition Using Smartphones dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). It uses the dataset's 561-feature vectors to classify six activities. Each simulated client is assigned the official training rows for exactly one participant. The server evaluates the global model against the official test split.

## Set up the project

Flower's app bundle excludes the dataset's `.txt` files. The absolute path to the existing dataset directory is configured as `dataset-dir` in `pyproject.toml`. The server and each client read the files directly from that location without copying or downloading them.

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` and the project:

```bash
pip install -e .
```

## Run the project

The project can run in Flower's simulation and deployment modes. Simulation is the simplest way to get started.

### Run with the Simulation Engine

Choose the number of clients when launching the simulation. For example, to train with five clients:

```bash
flwr run . --federation-config="num-supernodes=5" --stream
```

Each client is assigned the training data of exactly one participant, selected by sorted `subject_train.txt` ID. The number of clients can range from 1 to 21.

When fewer than 21 clients are selected, only the corresponding number of participants is used for training. Each participant is assigned to at most one client, and participants are not combined within a client. The official test split remains separate and is used for server-side evaluation.

To train with three clients:

```bash
flwr run . --federation-config="num-supernodes=3" --stream
```

To train with all 21 training participants:

```bash
flwr run . --federation-config="num-supernodes=21" --stream
```

Run settings such as the number of server rounds and the learning rate can be overridden:

```bash
flwr run . --federation-config="num-supernodes=5" --run-config="num-server-rounds=5 learning-rate=0.05" --stream
```

In this example, the simulation uses five clients, runs for five server rounds, and sets the learning rate to `0.05`.

For more information, see the [Flower PyTorch quickstart tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html).
