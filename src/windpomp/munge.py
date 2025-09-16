import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import time, sys, traceback

device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
if torch.cuda.is_available():
    device = "cuda"

model, preprocess = clip.load("ViT-B/32", device=device)


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None

    images, paths = zip(*batch)
    return torch.stack(images), paths


class ImageFolderDataset(Dataset):
    def __init__(self, images_list: list[str], preprocess):
        self.image_files = images_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        if idx % 1000 == 0:
            print(f"{time.ctime()} at {idx} with {image_path}")
        try:
            image = Image.open(image_path).convert("RGBA")
        except:
            print(traceback.format_exc())
            return None
        return self.preprocess(image), image_path  # return both tensor + path


def process_imagelist(images: list[str], out_filename: str):
    dataset = ImageFolderDataset(images, preprocess)
    dataloader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=8, collate_fn=collate_fn
    )

    all_features = []
    all_paths = []

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            features = model.encode_image(images)

            # Normalize (optional, useful for similarity search)
            features = features / features.norm(dim=-1, keepdim=True)

            all_features.append(features.cpu())
            all_paths.extend(paths)

    all_features = torch.cat(all_features, dim=0)
    features_np = all_features.numpy()

    df = pd.DataFrame(
        {
            "path": all_paths,
            "features": list(features_np),
        }  # each row is a 1D numpy array
    )

    df["features"] = df["features"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
    df.to_parquet(out_filename, index=False)


if __name__ == "__main__":
    import_path = sys.argv[1]
    if not os.path.isdir(import_path):
        raise ValueError(f"Provided path {import_path} is not a directory")
    else:
        images = [os.path.join(import_path, x) for x in os.listdir(import_path)]
        print(f"Read {len(images)} from {import_path}")
    process_imagelist(images, f"{time.time()}.parquet")
