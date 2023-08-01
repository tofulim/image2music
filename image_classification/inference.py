import argparse
import os
import urllib.request
from io import BytesIO

import requests
from PIL import Image
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from models.image_cls_model import ImageClassificationModel


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = ImageClassificationModel(num_labels=147).eval().to(device)
train_df = pd.read_csv("data/train_data.csv")
valid_df = pd.read_csv("data/valid_data.csv")
total_playlists_df = pd.concat([train_df, valid_df]).reset_index(drop=True)

preprocessor = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def inference(model: ImageClassificationModel, image_url: str):
    with torch.no_grad():
        req = urllib.request.Request(
            image_url, headers={"User-Agent": "Mozilla/5.0"}
        )
        res = urllib.request.urlopen(req).read()
        input_image = Image.open(BytesIO(res))
        input_image = preprocessor(input_image).to(device)
        print(input_image.shape)
        input_image = input_image.unsqueeze(0)
        print(input_image.shape)
        outputs = model(input_image)
        prediction = outputs.argmax(dim=-1).tolist()
        print(f"prediction : {prediction}")
    return prediction


def pred2playlist(prediction: int, playlists: pd.DataFrame):
    """
    모델 예측을 통해 구한 라벨 인덱스로 playlists 중 playlist를 선택하고 노래들 중 하나를 무작위로 선곡한다.

    Args:
        prediction (int): playlists의 index
        playlists (pd.DataFrame): 여러 vid_label를 가진 playlist로 이루어진 DataFrame

    Returns:
        song (tuple): 하나의 노래에 대한 정보

    """
    song = playlists.query("vid_label==@prediction").sample(
        n=1
    )

    return song


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="image2song")
    parser.add_argument("--image_url", type=str, required=True)

    args = parser.parse_args()
    image_url = args.image_url

    prediction = inference(model=model, image_url=image_url)
    song = pred2playlist(prediction=prediction, playlists=total_playlists_df)

    print(song)
