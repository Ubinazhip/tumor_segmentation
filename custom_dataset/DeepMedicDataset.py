from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import torch
import os
import random
from dpipe.io import load_numpy
from pathlib import Path
import pandas as pd
import torch.nn as nn
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
class DeepMedicDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, source_folder: [str, Path], nonzero_mask=False, transform1=None, transform2=None,
                 crop_size=(160, 160)):
        if isinstance(source_folder, str):
            source_folder = Path(source_folder)
        self.crop_size = crop_size
        if nonzero_mask:
            meta = meta[meta.sample_id.isin(
                meta.query('is_nonzero_mask == True').sample_id)]

        self.meta_flair = meta.loc[meta['img_type'] == 'flair']
        self.source_folder = source_folder

        self.meta_images_flair = self.meta_flair.query(
            'is_mask == False').sort_values(by='sample_id').reset_index(drop=True)
        self.meta_images = meta.query('is_mask == False').sort_values(
            by='sample_id').reset_index(drop=True)
        self.meta_masks = meta.query('is_mask == True').sort_values(
            by='sample_id').reset_index(drop=True)
        self.transform1 = transform1
        self.transform2 = transform2
    def __len__(self):
        return self.meta_images_flair.shape[0]

    def preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask[mask == False] = 0.0
        mask[mask == True] = 1.0
        return mask
    def __getitem__(self, i):

        relative_path = self.meta_images_flair.iloc[i]['relative_path']

        sample_id = self.meta_images_flair.loc[self.meta_images_flair['relative_path']
                                               == relative_path]['sample_id'][i]
        # print(f'sample_id is {sample_id}')
        subject_id = self.meta_images_flair.loc[self.meta_images_flair['relative_path']
                                                == relative_path]['subject_id'][i]

        img_t1_path = self.meta_images.loc[(self.meta_images['sample_id'] == sample_id) & (
                self.meta_images['img_type'] == 't1')]['relative_path'].reset_index(drop=True)[0]
        img_t1ce_path = self.meta_images.loc[(self.meta_images['sample_id'] == sample_id) & (
                self.meta_images['img_type'] == 't1ce')]['relative_path'].reset_index(drop=True)[0]
        img_t2_path = self.meta_images.loc[(self.meta_images['sample_id'] == sample_id) & (
                self.meta_images['img_type'] == 't2')]['relative_path'].reset_index(drop=True)[0]

        # , decompress=True)
        img_flair = load_numpy(
            self.source_folder / relative_path, allow_pickle=True)
        img_t1 = load_numpy(self.source_folder /
                            img_t1_path, allow_pickle=True)
        img_t1ce = load_numpy(self.source_folder /
                              img_t1ce_path, allow_pickle=True)
        img_t2 = load_numpy(self.source_folder /
                            img_t2_path, allow_pickle=True)

        # , decompress=True)
        mask = load_numpy(
            self.source_folder / self.meta_masks.iloc[i]['relative_path'], allow_pickle=True)
        img_concat = np.concatenate((img_t1[:,:,None], img_t1ce[:,:,None], img_t2[:,:,None], img_flair[:,:,None]), axis = 2)
        mask = self.preprocess_mask(mask)
        if self.transform1:
            augmented = self.transform1(image=img_concat, mask=mask)
            img_aug = augmented['image']
            mask = augmented['mask']
        if self.transform2:
            augmented = self.transform2(image =img_aug)
            high_resolution = augmented['image']
           # high_resolution = high_resolution[None,:]
        transform =A.Compose([
            A.Resize(height=104, width=104, always_apply=True),
           # ToTensorV2()
            ])
        mask = transform(image=mask)['image']
        transform = A.Compose([
            # A.CenterCrop(p=1, height=104, width=104),
            A.Resize(height=40, width=40, always_apply=True),
            #ToTensorV2()
        ])

        low_resolution = transform(image=img_aug)['image']
        high_resolution = np.einsum('abc->cab', high_resolution)
        low_resolution = np.einsum('abc->cab', low_resolution)
        mask = mask[None,:]
     #   print(f'hr {high_resolution.shape}')
     #   print(f'lw {low_resolution.shape}')
     #   print(f'mask = {mask.shape}')


        return high_resolution, low_resolution, mask, subject_id


class DeepMedicDataset_old(Dataset):
    def __init__(self, meta: pd.DataFrame, source_folder: [str, Path], nonzero_mask=False, transform1=None, transform2=None,
                 crop_size=(160, 160)):
        if isinstance(source_folder, str):
            source_folder = Path(source_folder)
        self.crop_size = crop_size
        if nonzero_mask:
            meta = meta[meta.sample_id.isin(
                meta.query('is_nonzero_mask == True').sample_id)]

        self.meta_flair = meta.loc[meta['img_type'] == 'flair']
        self.source_folder = source_folder

        self.meta_images_flair = self.meta_flair.query(
            'is_mask == False').sort_values(by='sample_id').reset_index(drop=True)
        self.meta_images = meta.query('is_mask == False').sort_values(
            by='sample_id').reset_index(drop=True)
        self.meta_masks = meta.query('is_mask == True').sort_values(
            by='sample_id').reset_index(drop=True)
        self.transform1 = transform1
        self.transform2 = transform2
    def __len__(self):
        return self.meta_images_flair.shape[0]

    def preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask[mask == False] = 0.0
        mask[mask == True] = 1.0
        return mask
    def __getitem__(self, i):

        relative_path = self.meta_images_flair.iloc[i]['relative_path']

        sample_id = self.meta_images_flair.loc[self.meta_images_flair['relative_path']
                                               == relative_path]['sample_id'][i]
        # print(f'sample_id is {sample_id}')
        subject_id = self.meta_images_flair.loc[self.meta_images_flair['relative_path']
                                                == relative_path]['subject_id'][i]

        img_t1_path = self.meta_images.loc[(self.meta_images['sample_id'] == sample_id) & (
                self.meta_images['img_type'] == 't1')]['relative_path'].reset_index(drop=True)[0]
        img_t1ce_path = self.meta_images.loc[(self.meta_images['sample_id'] == sample_id) & (
                self.meta_images['img_type'] == 't1ce')]['relative_path'].reset_index(drop=True)[0]
        img_t2_path = self.meta_images.loc[(self.meta_images['sample_id'] == sample_id) & (
                self.meta_images['img_type'] == 't2')]['relative_path'].reset_index(drop=True)[0]

        # , decompress=True)
        img_flair = load_numpy(
            self.source_folder / relative_path, allow_pickle=True)
        img_t1 = load_numpy(self.source_folder /
                            img_t1_path, allow_pickle=True)
        img_t1ce = load_numpy(self.source_folder /
                              img_t1ce_path, allow_pickle=True)
        img_t2 = load_numpy(self.source_folder /
                            img_t2_path, allow_pickle=True)

        # , decompress=True)
        mask = load_numpy(
            self.source_folder / self.meta_masks.iloc[i]['relative_path'], allow_pickle=True)
        img_concat = np.concatenate((img_t1[:,:,None], img_t1ce[:,:,None], img_t2[:,:,None], img_flair[:,:,None]), axis = 2)
        mask = self.preprocess_mask(mask)
        if self.transform1:
            augmented = self.transform1(image=img_concat, mask=mask)
            img_aug = augmented['image']
            mask = augmented['mask']
        if self.transform2:
            augmented = self.transform2(image =img_aug)
            high_resolution = augmented['image']
           # high_resolution = high_resolution[None,:]
        transform =A.Compose([
            A.Resize(height=104, width=104, always_apply=True),
            ToTensorV2()
            ])
        mask = transform(image=mask)['image']
        transform = A.Compose([
            # A.CenterCrop(p=1, height=104, width=104),
            A.Resize(height=40, width=40, always_apply=True),
            ToTensorV2()
        ])

        low_resolution = transform(image=img_aug)['image']


        return high_resolution, low_resolution, mask, subject_id


class DeepMedicDataset_old2(Dataset):
    def __init__(self, meta: pd.DataFrame, source_folder: [str, Path], nonzero_mask=False, crop_size=(200, 200), transform1=None,transform2=None, resize=None):
        if isinstance(source_folder, str):
            source_folder = Path(source_folder)
        self.crop_size = crop_size
        if nonzero_mask:
            meta = meta[meta.sample_id.isin(
                meta.query('is_nonzero_mask == True').sample_id)]

        self.meta_flair = meta.loc[meta['img_type'] == 'flair']
        self.source_folder = source_folder

        self.meta_images_flair = self.meta_flair.query(
            'is_mask == False').sort_values(by='sample_id').reset_index(drop=True)
        self.meta_images = meta.query('is_mask == False').sort_values(
            by='sample_id').reset_index(drop=True)
        self.meta_masks = meta.query('is_mask == True').sort_values(
            by='sample_id').reset_index(drop=True)
     #   self.transform = transform

    def __len__(self):
        return self.meta_images_flair.shape[0]

    def __getitem__(self, i):

        relative_path = self.meta_images_flair.iloc[i]['relative_path']

        sample_id = self.meta_images_flair.loc[self.meta_images_flair['relative_path']
                                               == relative_path]['sample_id'][i]
        # print(f'sample_id is {sample_id}')
        subject_id = self.meta_images_flair.loc[self.meta_images_flair['relative_path']
                                                == relative_path]['subject_id'][i]

        img_t1_path = self.meta_images.loc[(self.meta_images['sample_id'] == sample_id) & (
                self.meta_images['img_type'] == 't1')]['relative_path'].reset_index(drop=True)[0]
        img_t1ce_path = self.meta_images.loc[(self.meta_images['sample_id'] == sample_id) & (
                self.meta_images['img_type'] == 't1ce')]['relative_path'].reset_index(drop=True)[0]
        img_t2_path = self.meta_images.loc[(self.meta_images['sample_id'] == sample_id) & (
                self.meta_images['img_type'] == 't2')]['relative_path'].reset_index(drop=True)[0]

        # , decompress=True)
        img_flair = load_numpy(
            self.source_folder / relative_path, allow_pickle=True)
        img_t1 = load_numpy(self.source_folder /
                            img_t1_path, allow_pickle=True)
        img_t1ce = load_numpy(self.source_folder /
                              img_t1ce_path, allow_pickle=True)
        img_t2 = load_numpy(self.source_folder /
                            img_t2_path, allow_pickle=True)

        # , decompress=True)
        mask = load_numpy(
            self.source_folder / self.meta_masks.iloc[i]['relative_path'], allow_pickle=True)
        sample = img_flair, mask
       # if self.transform:
       #     image, mask = self.transform(sample)
        img_flair = torch.from_numpy(img_flair).reshape(1, 240, 240)
        img_t1 = torch.from_numpy(img_t1).reshape(1, 240, 240)
        img_t1ce = torch.from_numpy(img_t1ce).reshape(1, 240, 240)
        img_t2 = torch.from_numpy(img_t2).reshape(1, 240, 240)
        mask = torch.from_numpy(mask).reshape(1, 240, 240).double()

        img_flair = (img_flair - img_flair.mean()) / img_flair.std()
        img_t1 = (img_t1 - img_t1.mean()) / img_t1.std()
        img_t1ce = (img_t1ce - img_t1ce.mean()) / img_t1ce.std()
        img_t2 = (img_t2 - img_t2.mean()) / img_t2.std()

        delta1 = np.random.randint(10, 240 - self.crop_size[0] - 10)
        delta2 = np.random.randint(10, 240 - self.crop_size[1] - 10)
        img_flair_crop = img_flair[
                         ...,
                         delta1: delta1 + self.crop_size[0],
                         delta2: delta2 + self.crop_size[1]
                         ]

        img_t1_crop = img_t1[
                      ...,
                      delta1: delta1 + self.crop_size[0],
                      delta2: delta2 + self.crop_size[1]
                      ]

        img_t1ce_crop = img_t1ce[
                        ...,
                        delta1: delta1 + self.crop_size[0],
                        delta2: delta2 + self.crop_size[1]
                        ]
        img_t2_crop = img_t2[
                      ...,
                      delta1: delta1 + self.crop_size[0],
                      delta2: delta2 + self.crop_size[1]
                      ]
        mask_crop = mask[
                    ...,
                    delta1: delta1 + self.crop_size[0],
                    delta2: delta2 + self.crop_size[1]
                    ]
        img_concat = torch.cat(
            (img_t1_crop, img_t1ce_crop, img_t2_crop, img_flair_crop), dim=0)
        # high resolution
        crop_hr = 80  # 160
        crop_lr = 100
        crop_mask = 72  # 84
        hr = img_concat[...,
             int(img_concat.shape[1] / 2 - crop_hr):int(img_concat.shape[1] / 2 + crop_hr),
             int(img_concat.shape[2] / 2 - crop_hr): int(img_concat.shape[2] / 2 + crop_hr)
             ]
        # low resolution
        lw = torch.nn.functional.interpolate(img_concat[None,:], crop_lr)
        mask_crop = mask_crop[
                    ...,
                    int(mask_crop.shape[2] / 2) - crop_mask: int(mask_crop.shape[2] / 2) + crop_mask,
                    int(mask_crop.shape[2] / 2) - crop_mask: int(mask_crop.shape[2] / 2) + crop_mask
                    ]
        #print(f'lw before is {lw.shape}')
        lw = lw .squeeze(1)
        #print(f'after is {lw.shape}')
        #print(hr.shape)
        return hr, lw[0], mask_crop, subject_id