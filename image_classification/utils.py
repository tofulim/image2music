import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def log_metrics(
    preds: list,
    labels: list,
    loss: float,
    prefix: str,
    step: int,
):
    """
    (train | valid)로부터 받은 예측, 정답 데이터를 통해 지표를 계산하고 반환한다.

    Args:
        preds (list):
        labels (list):
        loss (float):
        prefix (str):
        step (int):

    Returns:
        metrics (dict): f1, acc, recall, precision을 담은 dict

    """

    metrics = {
        f"{prefix}_f1_score": f1_score(labels, preds, average="macro"),
        f"{prefix}_accuracy_score": accuracy_score(labels, preds),
        f"{prefix}_recall_score": recall_score(labels, preds, average="macro"),
        f"{prefix}_precision_score": precision_score(labels, preds, average="macro"),
        f"{prefix}_loss": loss,
    }

    wandb.log(metrics, step=step)

    return metrics
