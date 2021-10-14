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
from torchvision import transforms
import albumentations as A
class BraTSDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, source_folder: [str, Path], nonzero_mask=False, transform1=None, transform2=None):
        if isinstance(source_folder, str):
            source_folder = Path(source_folder)

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
        #print(f'sample_id is {sample_id}')
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

        img_concat = np.concatenate(
            (img_t1[:, :, None], img_t1ce[:, :, None], img_t2[:, :, None], img_flair[:, :, None]), axis=2)
        mask = self.preprocess_mask(mask)

        if self.transform1:
            augmented = self.transform1(image=img_concat, mask=mask)

            img_concat = augmented['image']
            mask = augmented['mask']
        
        img_concat = np.einsum('abc->cab', img_concat)
        #mask = torch.from_numpy(mask)
        return img_concat, mask[None,:], subject_id


class BraTSDataset_old(Dataset):
    def __init__(self, meta: pd.DataFrame, source_folder: [str, Path], nonzero_mask=False, transform1=None, transform2=None):
        if isinstance(source_folder, str):
            source_folder = Path(source_folder)

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

    def __getitem__(self, i):


        relative_path = self.meta_images_flair.iloc[i]['relative_path']

        sample_id = self.meta_images_flair.loc[self.meta_images_flair['relative_path']
                                               == relative_path]['sample_id'][i]
        #print(f'sample_id is {sample_id}')
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

        print(img_t2)
        #transform = transforms.Resize(size= (512, 512))
        #img_flair = torch.from_numpy(img_flair).reshape(1, 240, 240)
        #img_t1 = torch.from_numpy(img_t1).reshape(1, 240, 240)
        #img_t1ce = torch.from_numpy(img_t1ce).reshape(1, 240, 240)
        #img_t2 = torch.from_numpy(img_t2).reshape(1, 240, 240)
        #mask = torch.from_numpy(mask).reshape(1, 240, 240).double()
        #sample = img_flair, img_t1, img_t1ce, img_t2, mask
       # img_flair, img_t1, img_t1ce, img_t2, mask = torch.nn.functional.interpolate(sample, size=512)
        #img_flair = torch.nn.functional.interpolate(img_flair, size=512)
        #print(img_flair.shape)

        img_concat = torch.cat(
                (img_t1, img_t1ce, img_t2, img_flair), dim=0)
        return img_concat, mask, subject_id


class BraTSDataset_old2(Dataset):
    def __init__(self, meta: pd.DataFrame, source_folder: [str, Path], nonzero_mask=False,  transform1=None, transform2=None, resize = None):
        if isinstance(source_folder, str):
            source_folder = Path(source_folder)

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
        self.transform1 = transform1
        self.transform2 = transform2
        self.resize = resize
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
      #  if self.transform:
      #      image, mask = self.transform(sample)
        img_flair = torch.from_numpy(img_flair).reshape(1, 240, 240)
        img_t1 = torch.from_numpy(img_t1).reshape(1, 240, 240)
        img_t1ce = torch.from_numpy(img_t1ce).reshape(1, 240, 240)
        img_t2 = torch.from_numpy(img_t2).reshape(1, 240, 240)
        mask = torch.from_numpy(mask).reshape(1, 240, 240).double()

        img_concat = torch.cat(
            (img_t1, img_t1ce, img_t2, img_flair), dim=0)

        if self.resize:
            img_concat = torch.nn.functional.interpolate(img_concat[None,:], size=self.resize, mode='bilinear', align_corners=False)
            mask = torch.nn.functional.interpolate(mask[None,:], size=self.resize, mode='bilinear', align_corners=False)
            return img_concat[0], mask[0], subject_id
        else:
            return img_concat, mask, subject_id