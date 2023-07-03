import os

import torch
from torch import optim
from tqdm import tqdm
from utils import log_metrics


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        num_epochs,
        device,
        save_dir: str = "ckpt",
    ) -> None:
        super(Trainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_every = len(self.train_loader) // 5
        self.log_every = len(self.train_loader) // 20
        self.optimizer = optim.Adam(model.parameters(), lr=3e-4)
        self.num_epochs = num_epochs
        self.device = device
        self.max_valid_f1_score = 0.0
        self.min_valid_loss = 100000000
        self.save_dir = save_dir
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.global_step = 0

    def _train(self, epoch: int):
        total_loss, train_predictions, train_labels = 0.0, [], []

        self.model.train()
        for step, items in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()

            input_images, labels = items["images"], items["labels"]
            input_images = input_images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_images)
            loss = self.loss_fn(outputs, labels)

            loss = loss.mean()
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            train_predictions.extend(outputs.argmax(dim=-1).tolist())
            train_labels.extend(labels.tolist())

            # log train score
            if self.global_step != 0 and self.global_step % self.log_every == 0:
                _ = log_metrics(
                    preds=train_predictions,
                    labels=train_labels,
                    loss=total_loss / (step + 1),
                    prefix="train",
                    step=self.global_step,
                )
                train_predictions, train_labels = [], []

            # log and evaluate valid score
            if self.global_step != 0 and self.global_step % self.valid_every == 0:
                valid_loss, valid_metrics = self._evaluate(self.valid_loader, "valid")
                if valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    dir_name = f"{epoch}_step{self.global_step}"
                    if os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    torch.save(
                        self.model.state_dict(), os.path.join(self.save_dir, dir_name)
                    )
                    print(f"best ckpt in ep{epoch} step{self.global_step}!")

            self.global_step += step + 1

        total_loss /= len(self.train_loader)
        print(f"epoch train loss is {total_loss}")

    def _evaluate(self, valid_loader, prefix):
        targets = list()
        predictions = list()
        eval_loss = 0

        self.model.eval()
        with torch.no_grad():
            for step, items in enumerate(tqdm(valid_loader)):
                input_images, labels = items["images"], items["labels"]
                input_images = input_images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_images)
                loss = self.loss_fn(outputs, labels)

                loss = loss.mean()
                eval_loss += loss.item()

                targets.extend(labels.tolist())
                predictions.extend(outputs.argmax(dim=-1).tolist())

        metrics = log_metrics(
            preds=predictions,
            labels=labels,
            loss=eval_loss,
            prefix="train",
            step=self.global_step,
        )

        return eval_loss / len(valid_loader), metrics

    def fit(self):
        for epoch in range(self.num_epochs):
            self._train(epoch)
