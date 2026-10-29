"""Custom logging utilities for the UCI HAR Flower app."""

from flwr.serverapp.strategy import FedAvg


class LoggingFedAvg(FedAvg):
    """FedAvg with readable per-client and global metric summaries."""

    def __init__(self, total_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.total_rounds = total_rounds
        self._client_labels: dict[int, str] = {}
        self._printed_node_mappings: set[int] = set()

    def _client_label(self, node_id: int) -> str:
        """Assign a stable short label to each Flower node ID."""
        if node_id not in self._client_labels:
            self._client_labels[node_id] = (
                f"Client {len(self._client_labels) + 1:02d}"
            )
        return self._client_labels[node_id]

    @staticmethod
    def _fmt(value, digits: int = 4) -> str:
        return f"{float(value):.{digits}f}" if value is not None else "—"

    def aggregate_train(self, server_round, replies):
        # Materialize replies so they can be inspected before FedAvg aggregates them.
        replies = list(replies)
        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        successful = []
        new_mappings = []
        failed = 0

        for reply in replies:
            if reply.has_error():
                failed += 1
                continue

            if not reply.has_content() or "metrics" not in reply.content:
                failed += 1
                continue

            metrics = reply.content["metrics"]
            node_id = reply.metadata.src_node_id
            label = self._client_label(node_id)

            if node_id not in self._printed_node_mappings:
                new_mappings.append((label, node_id))
                self._printed_node_mappings.add(node_id)

            successful.append(
                (
                    label,
                    int(metrics.get("num-examples", 0)),
                    metrics.get("train_loss"),
                )
            )

        line = "=" * 78
        print(f"\n{line}")
        print(f"ROUND {server_round:02d}  |  LOCAL TRAINING")
        print("-" * 78)
        print(f"{'CLIENT':<14}{'EXAMPLES':>12}{'TRAIN LOSS':>16}")
        print("-" * 78)

        for label, num_examples, train_loss in sorted(successful):
            print(
                f"{label:<14}"
                f"{num_examples:>12,}"
                f"{self._fmt(train_loss):>16}"
            )

        aggregated_loss = (
            aggregated_metrics.get("train_loss")
            if aggregated_metrics is not None
            else None
        )

        print("-" * 78)
        print(
            f"Replies: {len(successful)}/{len(replies)} successful  |  "
            f"Failures: {failed}  |  "
            f"Weighted train loss: {self._fmt(aggregated_loss)}"
        )

        if new_mappings:
            print(
                "Client ID mapping: "
                + "  |  ".join(
                    f"{label} = node {node_id}"
                    for label, node_id in new_mappings
                )
            )

        return aggregated_arrays, aggregated_metrics

    def aggregate_evaluate(self, server_round, replies):
        # Materialize replies so they can be inspected before FedAvg aggregates them.
        replies = list(replies)
        client_rows = []
        new_mappings = []
        failed = 0

        for reply in replies:
            if reply.has_error():
                failed += 1
                print(
                    f"Client node {reply.metadata.src_node_id}: "
                    f"evaluation failed ({reply.error})"
                )
                continue

            if not reply.has_content() or "metrics" not in reply.content:
                failed += 1
                print(
                    f"Client node {reply.metadata.src_node_id}: "
                    "no evaluation metrics returned"
                )
                continue

            metrics = reply.content["metrics"]
            node_id = reply.metadata.src_node_id
            label = self._client_label(node_id)

            if node_id not in self._printed_node_mappings:
                new_mappings.append((label, node_id))
                self._printed_node_mappings.add(node_id)

            client_rows.append(
                {
                    "client": label,
                    "node_id": node_id,
                    "examples": int(metrics.get("num-examples", 0)),
                    "loss": metrics.get("eval_loss"),
                    "accuracy": metrics.get("eval_acc"),
                    "precision": metrics.get("eval_precision"),
                    "recall": metrics.get("eval_recall"),
                    "f1": metrics.get("eval_f1"),
                }
            )

        line = "=" * 78
        print(f"\n{line}")
        print(f"ROUND {server_round:02d}  |  CLIENT VALIDATION")
        print("-" * 78)
        print(
            f"{'CLIENT':<14}{'N':>9}{'LOSS':>11}{'ACCURACY':>12}"
            f"{'MACRO-P':>11}{'MACRO-R':>11}{'MACRO-F1':>11}"
        )
        print("-" * 78)

        for row in sorted(client_rows, key=lambda item: item["client"]):
            accuracy = (
                f"{float(row['accuracy']):.2%}"
                if row["accuracy"] is not None
                else "—"
            )

            print(
                f"{row['client']:<14}"
                f"{row['examples']:>9,}"
                f"{self._fmt(row['loss']):>11}"
                f"{accuracy:>12}"
                f"{self._fmt(row['precision']):>11}"
                f"{self._fmt(row['recall']):>11}"
                f"{self._fmt(row['f1']):>11}"
            )

        accuracies = [
            float(row["accuracy"])
            for row in client_rows
            if row["accuracy"] is not None
        ]

        if accuracies:
            mean_acc = sum(accuracies) / len(accuracies)
            min_acc = min(accuracies)
            max_acc = max(accuracies)
            gap_pp = (max_acc - min_acc) * 100

            print("-" * 78)
            print(
                "Client accuracy (unweighted)  |  "
                f"Mean: {mean_acc:.2%}  |  "
                f"Worst: {min_acc:.2%}  |  "
                f"Best: {max_acc:.2%}  |  "
                f"Gap: {gap_pp:.2f} pp"
            )

        print(
            f"Replies: {len(client_rows)}/{len(replies)} successful  |  "
            f"Failures: {failed}"
        )

        if new_mappings:
            print(
                "Client ID mapping: "
                + "  |  ".join(
                    f"{label} = node {node_id}"
                    for label, node_id in new_mappings
                )
            )

        return super().aggregate_evaluate(server_round, replies)