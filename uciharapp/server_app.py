"""UCI HAR Flower server application."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from uciharapp.logging_utils import LoggingFedAvg

from uciharapp.task import (
    Net,
    load_centralized_dataset,
    resolve_dataset_dir,
    test_metrics,
)

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    dataset_dir = str(resolve_dataset_dir(context.run_config["dataset-dir"]))
    print(f"Using UCI HAR dataset at {dataset_dir}")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with concise custom logs
    strategy = LoggingFedAvg(
        fraction_evaluate=fraction_evaluate,
        total_rounds=num_rounds,
    )

    print("\nStarting federated training")
    print(f"Rounds: {num_rounds}  |  Evaluation fraction: {fraction_evaluate:.0%}")

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=lambda server_round, arrays: global_evaluate(
            server_round, arrays, dataset_dir
        ),
    )

    if context.run_config["save-model"]:
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")
        print("\nFinal model saved: final_model.pt")

    print("\nFederated training finished.")


def global_evaluate(
    server_round: int, arrays: ArrayRecord, dataset_dir: str | None = None
) -> MetricRecord:
    """Evaluate the global model and print a compact metric summary."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the centralized dataset
    test_dataloader = load_centralized_dataset(dataset_dir)

    # Evaluate the global model centrally
    metrics = test_metrics(model, test_dataloader, device)
    phase = "BASELINE (BEFORE TRAINING)" if server_round == 0 else "GLOBAL EVALUATION"

    print(f"\n{'-' * 78}")
    if server_round == 0:
        print(f"{phase}  |  INITIAL MODEL")
    else:
        print(f"ROUND {server_round:02d}  |  {phase}")

    print("-" * 78)
    print(
        f"Loss: {metrics['eval_loss']:.4f}  |  "
        f"Accuracy: {metrics['eval_acc']:.2%}  |  "
        f"Macro precision: {metrics['eval_precision']:.4f}  |  "
        f"Macro recall: {metrics['eval_recall']:.4f}  |  "
        f"Macro-F1: {metrics['eval_f1']:.4f}"
    )

    # Keep all metrics in the MetricRecord for Flower's result object.
    return MetricRecord(
        {
            "accuracy": metrics["eval_acc"],
            "loss": metrics["eval_loss"],
            "precision": metrics["eval_precision"],
            "recall": metrics["eval_recall"],
            "f1": metrics["eval_f1"],
            "confidence": metrics["eval_confidence"],
            "correct_confidence": metrics["eval_correct_confidence"],
            "wrong_confidence": metrics["eval_wrong_confidence"],
        }
    )
