from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pydicom import dcmread
from pydicom.pixels import apply_voi_lut
import numpy as np
import ast
import pandas as pd
import os
import torch
import cv2
from monai.transforms import *
from collections import Counter
from glob import glob


class VindrDicomDataset(Dataset):
    """Vindr Mammo custom PyTorch Dataset"""

    def __init__(self,
                 root,
                 class_list=['no_finding', 'suspicious_calcification', 'mass'],
                 image_size=512,
                 train=True):
        """_summary_

        Args:
            root (str): Path to vindr mammo extracted files
            class_list (list, optional): List of classes to keep in the dataset. Defaults to ['no_finding', 'suspicious_calcification', 'mass'].
            image_size (int, optional): Value to resize the images. Defaults to 512.
            train (bool, optional): Boolean to load the train or test split. Defaults to True.
        """
        self.train = train
        self.class_list = class_list
        self.data_dir = root
        self.df = self.__prepare_vindr_dataframe(root, class_list)
        self.targets = self.df['finding_categories'].values
        self.class_weights = self.__get_class_weights()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])

    def __get_class_weights(self):
        training_label_counts = Counter(self.targets)
        total_samples = sum(training_label_counts.values())
        class_weights = torch.tensor([total_samples / training_label_counts[cls]
                                      for cls in training_label_counts],
                                     dtype=torch.float
                                     ).to('cuda' if torch.cuda.is_available() else 'cpu')
        return class_weights

    def __load_dicom_image(self, path):
        ds = dcmread(path)
        img2d = ds.pixel_array
        img2d = apply_voi_lut(img2d, ds)

        if ds.PhotometricInterpretation == "MONOCHROME1":
            img2d = np.amax(img2d) - img2d

        img2d = img2d.astype(np.float32)
        normalized_data = (img2d - np.min(img2d)) / \
            (np.max(img2d) - np.min(img2d))
        return normalized_data

    def __crop_to_breast(self, original):
        img = (original * 255).astype('uint8')
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, mask = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cnts, _ = cv2.findContours(
            mask.astype(
                np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnt = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return original[y: y + h, x: x + w]

    def __prepare_vindr_dataframe(self, data_dir, class_list):

        def format_category_list(category_list):
            return [category.lower().replace(' ', '_') for category in category_list]

        def contains_all_classes(category_list, class_list):
            return any(cls in category_list for cls in class_list)

        def replace_categories(df, column, target_categories):
            def replace_if_present(categories):
                for target in target_categories:
                    if target in categories:
                        return target
                return categories

            df[column] = df[column].apply(
                lambda x: replace_if_present(x) if isinstance(x, list) else x)
            return df

        df_find = pd.read_csv(os.path.join(
            data_dir, 'finding_annotations.csv'))
        df_find['finding_categories'] = df_find['finding_categories'].apply(
            ast.literal_eval)
        df_find['finding_categories'] = df_find['finding_categories'].apply(
            format_category_list)
        df_find = df_find[df_find['finding_categories'].apply(
            lambda x: contains_all_classes(x, class_list))]
        df_find = replace_categories(df_find, 'finding_categories', class_list)
        split_name = 'training' if self.train else 'test'
        df_find = df_find[df_find['split'] == split_name]
        return df_find

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_path = os.path.join(
            self.data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')

        img = self.__load_dicom_image(sample_path)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = self.__crop_to_breast(img)
        img_tensor = self.transform(img)
        label = self.class_list.index(row['finding_categories'])
        return img_tensor, label


class CustomImageFolder(ImageFolder):
    def __init__(self, root, image_size: int = 256, train: bool = True):
        root = os.path.join(root, 'train' if train else 'test')
        super().__init__(root)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])
        self.class_weights = None
