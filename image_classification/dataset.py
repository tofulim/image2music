import random
import urllib.request
from io import BytesIO

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageDataset(Dataset):
    """
    youtube playlist thumbnail image dataset
    """

    def __init__(self, data_path: str):
        # 이미지 전처리 프로세서 로드
        self.processor = self._get_processor()
        data = pd.read_csv(data_path)

        self.images, self.labels = [], []
        # pd.DataFrame에 있는 image url 주소로 요청을 보내 이미지를 받아 저장한다.
        for vid_label, image_url in tqdm(data[["vid_label", "thumbnail_url"]].values):
            req = urllib.request.Request(
                image_url, headers={"User-Agent": "Mozilla/5.0"}
            )
            res = urllib.request.urlopen(req).read()
            input_image = Image.open(BytesIO(res))
            self.images.append(self.processor(input_image))
            self.labels.append(vid_label)

        self.images, self.labels = self._shuffle_items(self.images, self.labels)

    def _shuffle_items(self, *args):
        """
        주어진 데이터, 라벨 쌍을 섞어 반환한다.

        Args:
            *args (list): 섞어줄 images(list), labels(list)

        Returns:
            (images, labels)

        """
        items = list(zip(*args))
        random.shuffle(items)

        return list(zip(*items))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "images": self.images[idx],
            "labels": self.labels[idx],
        }

    def _get_processor(self, resize: int = 256):
        preprocess = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return preprocess


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_dataset = ImageDataset(data_path="image_classification/data/valid_data.csv")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
    )
    for td in train_dataloader:
        images = td["images"][0]
        label = td["labels"][0]
        print(f"image: {images}")
        print(f"label: {label}")
        break
