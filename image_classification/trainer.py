import os

import torch
import wandb
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
        self.valid_every = (
            len(self.train_loader) // 5 if len(self.train_loader) > 5 else 1
        )
        self.log_every = (
            len(self.train_loader) // 20 if len(self.train_loader) > 20 else 1
        )
        self.optimizer = optim.Adam(model.parameters(), lr=3e-4)
        self.num_epochs = num_epochs
        self.device = device
        self.max_valid_f1_score = 0.0
        self.min_valid_loss = 100000000
        self.save_dir = save_dir
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.global_step = 0

    def _train(self, epoch: int):
        total_loss = 0.0

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

            # log train score
            if self.global_step != 0 and self.global_step % self.log_every == 0:
                wandb.log(
                    {"train_loss": total_loss / (step + 1)}, step=self.global_step
                )

            # log and evaluate valid score
            if self.global_step != 0 and self.global_step % self.valid_every == 0:
                valid_loss, valid_metrics = self._evaluate(self.valid_loader, "valid")
                if valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    dir_name = f"{epoch}_step{self.global_step}"
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    torch.save(
                        self.model.state_dict(),
                        f"{os.path.join(self.save_dir, dir_name)}.pt",
                    )
                    print(f"best ckpt in ep{epoch} step{self.global_step}!")

            self.global_step += step + 1

        total_loss /= len(self.train_loader)
        print(f"epoch train loss is {total_loss}")

    def _evaluate(self, valid_loader, prefix):
        eval_labels = list()
        eval_predictions = list()
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

                eval_labels.extend(labels.tolist())
                eval_predictions.extend(outputs.argmax(dim=-1).tolist())

        eval_loss /= len(valid_loader)
        metrics = log_metrics(
            preds=eval_predictions,
            labels=eval_labels,
            loss=eval_loss,
            prefix=prefix,
            step=self.global_step,
        )

        return eval_loss, metrics

    def fit(self):
        for epoch in range(self.num_epochs):
            self._train(epoch)
