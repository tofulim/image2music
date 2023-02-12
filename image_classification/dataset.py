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
        self.processor = self._get_processor()
        data = pd.read_csv(data_path)

        self.images, self.labels = list(), list()

        for vid_label, image_url in tqdm(data[["vid_label", "thumbnail_url"]].values):
            req = urllib.request.Request(
                image_url, headers={"User-Agent": "Mozilla/5.0"}
            )
            res = urllib.request.urlopen(req).read()
            input_image = Image.open(BytesIO(res))
            self.images.append(self.processor(input_image))
            self.labels.append(vid_label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
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

    train_dataset = ImageDataset(data_path="data/youtube_data.csv")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
    )
    for td in train_dataloader:
        print(td)
        break
