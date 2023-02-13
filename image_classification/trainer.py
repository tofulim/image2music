import os

import torch
from torch import optim
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        scheduler,
        num_epochs,
        device,
        valid_every: int = 1_000,
        log_every: int = 20,
        save_dir: str = "ckpt",
    ) -> None:
        super(Trainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optim.Adam(model.parameters(), lr=3e-4)
        self.scheduler = scheduler
        self.valid_every = valid_every
        self.log_every = log_every
        self.num_epochs = num_epochs
        self.device = device
        self.max_valid_f1_score = 0.0
        self.min_valid_loss = 100000000
        self.save_dir = save_dir
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _train(self):
        total_loss = 0.0
        self.valid_every = len(self.train_loader) // 5

        self.model.train()

        for step, items in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()

            input_images, labels = items
            input_images = input_images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_images)
            loss = self.loss_fn(outputs, labels)

            loss = loss.mean()
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if step != 0 and step % self.valid_every == 0:
                valid_loss = self._evaluate(self.valid_loader, "valid")
                if valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    dir_name = f"step{step}"
                    if os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    self.model.module.save_pretrained(
                        os.path.join(self.save_dir, dir_name)
                    )
                    print(f"best ckpt in step{step} !")

    def _evaluate(self, valid_loader, prefix):
        targets = list()
        predictions = list()
        eval_loss = 0

        self.model.eval()
        with torch.no_grad():
            for step, items in enumerate(tqdm(valid_loader)):
                input_images, labels = items
                input_images = input_images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(input_images)
                loss = self.loss_fn(outputs, labels)

                loss = loss.mean()
                eval_loss += loss.item()

                targets.extend(labels.tolist())
                predictions.extend(outputs.logits.argmax(dim=-1).tolist())

        self.model.train()
        return eval_loss

    def fit(self):
        for epoch in range(self.num_epochs):
            self._train_epoch()
