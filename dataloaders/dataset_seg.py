import os
import pdb
import h5py
import torch
import itertools
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

def resampling(roiImg, new_size, lbl=False):
    new_spacing = [old_sz * old_spc / new_sz for old_sz, old_spc, new_sz in
                   zip(roiImg.GetSize(), roiImg.GetSpacing(), new_size)]
    if lbl:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    else:
        resampled_sitk = sitk.Resample(roiImg, new_size, sitk.Transform(), sitk.sitkLinear, roiImg.GetOrigin(),
                                       new_spacing, roiImg.GetDirection(), 0.0, roiImg.GetPixelIDValue())
    return resampled_sitk


class Pancreas(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir=None, num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform

        with open(self._base_dir+'/../train.list', 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + '/{}'.format(image_name), 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)
        image = (image - np.mean(image)) / np.std(image)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Pancreas_sparse(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir=None, num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform

        with open(self._base_dir+'/../train.list', 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

        self.slice_indices = []
        with open(self._base_dir+'/../labeled_slices.txt', 'r') as f:
            for line in f:
                sample_name, x_idx, y_idx, z_idx = line.strip().split(' ')
                self.slice_indices.append((sample_name, int(x_idx), int(y_idx), int(z_idx)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        sample_name, x_idx, y_idx, z_idx = self.slice_indices[idx]
        if image_name.split('.')[-2] != sample_name:
            print("eee")
        h5f = h5py.File(self._base_dir + '/{}'.format(image_name), 'r')
        image, label_full = h5f['image'][:], h5f['label'][:].astype(np.float32)
        image = (image - np.mean(image)) / np.std(image)
        label = np.zeros_like(label_full)
        label[x_idx, :, :] = label_full[x_idx, :, :]
        label[:, y_idx, :] = label_full[:, y_idx, :]
        label[:, :, z_idx] = label_full[:, :, z_idx]
        weight = np.zeros_like(label_full)
        weight[x_idx, :, :] = 1
        weight[:, y_idx, :] = 1
        weight[:, :, z_idx] = 1
        sample = {'image': image, 'label': label, 'weight': weight}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Pancreas_pseudo(Dataset):
    """ Pancreas Dataset """
    def __init__(self, base_dir=None, plabel_dir=None, num=None, transform=None):
        self._base_dir = base_dir
        self._plabel_dir = plabel_dir
        self.transform = transform

        with open(self._base_dir+'/../train.list', 'r') as f:
            self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

        self.slice_indices = []
        with open(self._base_dir+'/../labeled_slices.txt', 'r') as f:
            for line in f:
                sample_name, x_idx, y_idx, z_idx = line.strip().split(' ')
                self.slice_indices.append((sample_name, int(x_idx), int(y_idx), int(z_idx)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        sample_name, x_idx, y_idx, z_idx = self.slice_indices[idx]
        if image_name.split('.')[-2] != sample_name:
            print("eee")
        h5f = h5py.File(self._base_dir + '/{}'.format(image_name), 'r')
        image, label_full = h5f['image'][:], h5f['label'][:].astype(np.float32)
        image = (image - np.mean(image)) / np.std(image)
        plabel = nib.load(self._plabel_dir + '/{}_pseudo.nii.gz'.format(image_name.split('.')[-2])).get_fdata()
        plabel[x_idx, :, :] = label_full[x_idx, :, :]
        plabel[:, y_idx, :] = label_full[:, y_idx, :]
        plabel[:, :, z_idx] = label_full[:, :, z_idx]
        weight = np.zeros_like(label_full)
        weight[x_idx, :, :] = 1
        weight[:, y_idx, :] = 1
        weight[:, :, z_idx] = 1
        # sample = {'image': image, 'label': label_full, 'plabel': plabel}
        sample = {'image': image, 'label': plabel, 'weight': weight}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Sparse(object):
    def __call__(self, sample):
        image, label_full = sample['image'], sample['label']
        label = np.zeros_like(label_full)
        non_zero_slices_axis1 = np.where(np.any(label_full > 0, axis=(1, 2)))[0]
        non_zero_slices_axis2 = np.where(np.any(label_full > 0, axis=(0, 2)))[0]
        non_zero_slices_axis3 = np.where(np.any(label_full > 0, axis=(0, 1)))[0]
        slice1 = np.random.choice(non_zero_slices_axis1)
        slice2 = np.random.choice(non_zero_slices_axis2)
        slice3 = np.random.choice(non_zero_slices_axis3)
        label[slice1, :, :] = label_full[slice1, :, :]
        label[:, slice2, :] = label_full[:, slice2, :]
        label[:, :, slice3] = label_full[:, :, slice3]
        weight = np.zeros_like(label_full)
        weight[slice1, :, :] = 1
        weight[:, slice2, :] = 1
        weight[:, :, slice3] = 1
        return {'image': image, 'label': label, 'weight': weight}


class Resample(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if 'weight' in sample:
            image, label, weight = sample['image'], sample['label'], sample['weight']
            new_size = self.output_size
            image_itk = resampling(sitk.GetImageFromArray(image), new_size, lbl=False)
            label_itk = resampling(sitk.GetImageFromArray(label), new_size, lbl=True)
            weight_itk = resampling(sitk.GetImageFromArray(weight), new_size, lbl=True)
            image = sitk.GetArrayFromImage(image_itk)
            label = sitk.GetArrayFromImage(label_itk)
            weight = sitk.GetArrayFromImage(weight_itk)
            return {'image': image, 'label': label, 'weight': weight}

        else:
            image, label = sample['image'], sample['label']
            new_size = self.output_size
            image_itk = resampling(sitk.GetImageFromArray(image), new_size, lbl=False)
            label_itk = resampling(sitk.GetImageFromArray(label), new_size, lbl=True)
            image = sitk.GetArrayFromImage(image_itk)
            label = sitk.GetArrayFromImage(label_itk)
            if 'weight_2d' in sample:
                return {'image': image, 'label': label, 'weight_2d': sample['weight_2d']}
            else:
                return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        if 'weight' in sample:
            image, label, weight = sample['image'], sample['label'], sample['weight']
            # pad the sample if necessary
            if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
                pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
                pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                weight = np.pad(weight, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            (w, h, d) = image.shape
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            weight = weight[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'weight': weight}

        elif 'plabel' in sample:
            image, label, weight = sample['image'], sample['label'], sample['plabel']
            # pad the sample if necessary
            if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
                pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
                pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                weight = np.pad(weight, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            (w, h, d) = image.shape
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            weight = weight[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'plabel': weight}

        else:
            image, label = sample['image'], sample['label']
            # pad the sample if necessary
            if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                    self.output_size[2]:
                pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
                ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
                pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
                image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

            (w, h, d) = image.shape
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])
            d1 = np.random.randint(0, d - self.output_size[2])

            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __call__(self, sample):
        if 'weight' in sample:
            image, label, weight = sample['image'], sample['label'], sample['weight']
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            weight = np.rot90(weight, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            weight = np.flip(weight, axis=axis).copy()
            return {'image': image, 'label': label, 'weight': weight}
        
        elif 'plabel' in sample:
            image, label, weight = sample['image'], sample['label'], sample['plabel']
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            weight = np.rot90(weight, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            weight = np.flip(weight, axis=axis).copy()
            return {'image': image, 'label': label, 'plabel': weight}

        else:
            image, label = sample['image'], sample['label']
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            return {'image': image, 'label': label}


class RandomRotFlip_2d(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        if 'weight_2d' in sample:
            return {'image': image, 'label': label, 'weight_2d': sample['weight_2d']}
        else:
            return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        try:
            image = image.reshape(1, image.shape[0], image.shape[1]).astype(np.float32)
        except:
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'weight' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'weight': torch.from_numpy(sample['weight'])}
        elif 'weight_2d' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'weight_2d': sample['weight_2d']}
        elif 'plabel' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'plabel': torch.from_numpy(sample['plabel']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
