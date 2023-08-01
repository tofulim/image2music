import argparse
import random

import numpy as np
import pandas as pd
import torch
import wandb
from dataset import ImageDataset
from models.image_cls_model import ImageClassificationModel
from torch.utils.data import DataLoader
from trainer import Trainer


def _get_parser():
    parser = argparse.ArgumentParser(description="image2playlist")
    # Basic arguments
    parser.add_argument("--seed", type=int, default=118)
    parser.add_argument("--train_data_path", type=str, default="data/train_data.csv")
    parser.add_argument("--valid_data_path", type=str, default="data/valid_data.csv")
    parser.add_argument("--save_dir", type=str, required=True)
    # Training arguments
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-05)
    parser.add_argument("--num_epochs", type=int, default=10)
    return parser


def _init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    data_loaders = dict()

    parser = _get_parser()
    args = parser.parse_args()

    _init_seed(args.seed)
    wandb.init(project="youtube_playlist_classification", name="trial1")

    data_path_dict = {
        "train": args.train_data_path,
        "valid": args.valid_data_path,
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    for dataset_name, data_path in data_path_dict.items():
        dataset = ImageDataset(data_path=data_path)
        data_loaders[dataset_name] = DataLoader(
            dataset=dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            drop_last=False,
        )

    valid_df = pd.read_csv(args.valid_data_path)
    num_labels = len(valid_df["vid_label"].unique())
    model = ImageClassificationModel(num_labels=num_labels)
    model.to(device)

    trainer = Trainer(
        model=model,
        train_loader=data_loaders["train"],
        valid_loader=data_loaders["valid"],
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
