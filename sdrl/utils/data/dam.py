# coding: utf-8
"""
DAM (Disentanglement Aggregation Model) data module.
Swaps person and background between two images: person A on B's background, person B on A's background.
Uses pre-generated masks (*_mask.png) and optional precomputed backgrounds (*_bg.png).
"""
from __future__ import absolute_import
import os
import os.path as osp
import random
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

from .preprocessor import Preprocessor
from . import transforms as T


def _get_mask_path(img_path, mask_suffix="_mask.png"):
    """From image path get mask path: xxx.jpg -> xxx_mask.png (same dir)."""
    base, ext = osp.splitext(img_path)
    return base + mask_suffix


def _get_bg_path(img_path, bg_suffix="_bg.png"):
    """From image path get background path: xxx.jpg -> xxx_bg.png (same dir)."""
    base, ext = osp.splitext(img_path)
    return base + bg_suffix


def _load_image_array(path, root=None):
    if root:
        path = osp.join(root, path)
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _load_mask_binary(path, root=None, target_size=None):
    if root:
        path = osp.join(root, path)
    if not osp.isfile(path):
        return None
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    mask = (arr > 127).astype(np.float32)
    if target_size is not None:
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((target_size[1], target_size[0]), PILImage.NEAREST)
        mask = (np.array(mask_pil) > 127).astype(np.float32)
    return mask


def _composite_person_on_background(img_person, mask_person, img_bg, mask_bg, height, width):
    """
    Overlay person from (img_person, mask_person) onto background img_bg (inpainted, person region filled).
    mask_bg indicates where the person was in the background image (hole to paste into).
    All inputs are numpy (H,W,3) and (H,W), resized to (height, width) inside if needed.
    """
    from PIL import Image as PILImage
    h, w = height, width
    if img_person.shape[:2] != (h, w):
        img_person = np.array(PILImage.fromarray(img_person.astype(np.uint8)).resize((w, h), PILImage.BILINEAR))
    if img_bg.shape[:2] != (h, w):
        img_bg = np.array(PILImage.fromarray(img_bg.astype(np.uint8)).resize((w, h), PILImage.BILINEAR))
    if mask_person.shape[:2] != (h, w):
        mask_person = np.array(
            PILImage.fromarray((mask_person * 255).astype(np.uint8)).resize((w, h), PILImage.NEAREST)
        ).astype(np.float32) / 255.0
    if mask_bg.shape[:2] != (h, w):
        mask_bg = np.array(
            PILImage.fromarray((mask_bg * 255).astype(np.uint8)).resize((w, h), PILImage.NEAREST)
        ).astype(np.float32) / 255.0
    mask_person = np.expand_dims(mask_person, axis=2)
    mask_bg = np.expand_dims(mask_bg, axis=2)
    person_region = img_person.astype(np.float32) * mask_person
    out = img_bg.astype(np.float32) * (1.0 - mask_bg) + person_region * mask_bg
    return np.clip(out, 0, 255).astype(np.uint8)


class DAMPreprocessor(Dataset):
    """
    Dataset that returns either a normal sample or a DAM composite.
    item: int -> normal (img, fname, pid, camid, idx)
    item: (i, j, role) -> composite: role 0 = person i on bg j, role 1 = person j on bg i.
    Uses precomputed *_bg.png for inpainting; if missing, falls back to normal image for that slot.
    """
    def __init__(self, dataset, root=None, transform=None, height=384, width=128,
                 mask_suffix="_mask.png", bg_suffix="_bg.png", return_mask=False):
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.height = height
        self.width = width
        self.mask_suffix = mask_suffix
        self.bg_suffix = bg_suffix
        # When return_mask=True: apply synchronized geometric aug to image and person mask; also return mask (for SSDM Eq.19/24)
        self.return_mask = return_mask
        if return_mask:
            self._erase = T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
            self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def _fpath(self, fname):
        if self.root is not None:
            return osp.join(self.root, fname)
        return fname

    def _joint_transform(self, img_pil, mask_pil):
        """Apply synchronized geometric augmentation (resize/flip/pad/crop) to RGB image and grayscale mask,
        then ToTensor+Normalize(+RandomErasing) on image; binarize mask to [1,H,W].
        Matches the geometric part of the training transform."""
        img_pil = img_pil.resize((self.width, self.height), Image.BICUBIC)
        mask_pil = mask_pil.resize((self.width, self.height), Image.NEAREST)
        if random.random() < 0.5:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
        img_pil = ImageOps.expand(img_pil, border=10, fill=0)
        mask_pil = ImageOps.expand(mask_pil, border=10, fill=0)
        pw, ph = img_pil.size  # (width+20, height+20)
        x1 = random.randint(0, pw - self.width)
        y1 = random.randint(0, ph - self.height)
        img_pil = img_pil.crop((x1, y1, x1 + self.width, y1 + self.height))
        mask_pil = mask_pil.crop((x1, y1, x1 + self.width, y1 + self.height))

        img = torch.from_numpy(np.asarray(img_pil, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        img = (img - self._mean) / self._std
        img = self._erase(img)

        m = np.asarray(mask_pil, dtype=np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        mask = torch.from_numpy((m > 127).astype(np.float32).copy()).unsqueeze(0)
        return img, mask

    def _mask_pil_for(self, fpath, ref_size):
        """Load person mask (grayscale PIL); return all-ones if missing (equivalent to no masking). ref_size=(W,H)."""
        mask_path = _get_mask_path(fpath, self.mask_suffix)
        if osp.isfile(mask_path):
            mp = Image.open(mask_path).convert("L")
        else:
            mp = Image.new("L", ref_size, color=255)
        return mp

    def _mask_tensor_resize(self, mask_pil):
        """Resize mask only to training size; do not alter image augmentation path (preserves accuracy)."""
        mask_pil = mask_pil.resize((self.width, self.height), Image.NEAREST)
        m = np.asarray(mask_pil, dtype=np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]
        return torch.from_numpy((m > 127).astype(np.float32).copy()).unsqueeze(0)

    def _load_normal(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = self._fpath(fname)
        img_pil = Image.open(fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = img_pil
        if self.return_mask:
            mask_pil = self._mask_pil_for(fpath, img_pil.size)
            mask = self._mask_tensor_resize(mask_pil)
            return img, fname, pid, camid, index, mask
        return img, fname, pid, camid, index

    def _load_composite(self, i, j, role):
        # role 0: person i on background j; role 1: person j on background i
        fname_i, pid_i, camid_i = self.dataset[i]
        fname_j, pid_j, camid_j = self.dataset[j]
        fpath_i = self._fpath(fname_i)
        fpath_j = self._fpath(fname_j)
        mask_path_i = _get_mask_path(fpath_i, self.mask_suffix)
        mask_path_j = _get_mask_path(fpath_j, self.mask_suffix)
        bg_path_i = _get_bg_path(fpath_i, self.bg_suffix)
        bg_path_j = _get_bg_path(fpath_j, self.bg_suffix)

        if not osp.isfile(mask_path_i) or not osp.isfile(mask_path_j):
            # fallback: return normal image for person
            idx = i if role == 0 else j
            return self._load_normal(idx)
        if not osp.isfile(bg_path_i) or not osp.isfile(bg_path_j):
            idx = i if role == 0 else j
            return self._load_normal(idx)

        img_i = _load_image_array(fpath_i)
        img_j = _load_image_array(fpath_j)
        mask_i = _load_mask_binary(mask_path_i, root=None, target_size=None)
        mask_j = _load_mask_binary(mask_path_j, root=None, target_size=None)
        bg_i = _load_image_array(bg_path_i)
        bg_j = _load_image_array(bg_path_j)
        if mask_i is None or mask_j is None:
            idx = i if role == 0 else j
            return self._load_normal(idx)

        if role == 0:
            composite = _composite_person_on_background(
                img_i, mask_i, bg_j, mask_j, self.height, self.width
            )
            pid, camid, idx = pid_i, camid_j, i
            fname = fname_i
            paste_mask = mask_j  # Person (paste) region in the enhanced view
        else:
            composite = _composite_person_on_background(
                img_j, mask_j, bg_i, mask_i, self.height, self.width
            )
            pid, camid, idx = pid_j, camid_i, j
            fname = fname_j
            paste_mask = mask_i

        img = Image.fromarray(composite)
        if self.return_mask:
            mask_pil = Image.fromarray((paste_mask * 255).astype(np.uint8))
            if self.transform is not None:
                img = self.transform(img)
            mask = self._mask_tensor_resize(mask_pil)
            return img, fname, pid, camid, idx, mask
        if self.transform is not None:
            img = self.transform(img)
        return img, fname, pid, camid, idx

    def __getitem__(self, item):
        if isinstance(item, (list, tuple)):
            i, j, role = item[0], item[1], item[2]
            return self._load_composite(i, j, role)
        return self._load_normal(item)


class DAMBatchSampler(BatchSampler):
    """
    Wraps a sampler and yields batches of size batch_size (e.g. 32).
    Each batch: first half (16) are normal indices; second half (16) are 8 pairs of composite (i,j,0),(i,j,1).
    So we take 32 indices from the underlying sampler, use 0-15 as normal, and (16,17), (18,19), ... as 8 pairs.
    """
    def __init__(self, sampler, batch_size, drop_last=True):
        assert batch_size >= 4 and batch_size % 2 == 0
        super(DAMBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self._batch_size = batch_size

    def __iter__(self):
        batch = []
        half = self.batch_size // 2
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                out = []
                for k in range(half):
                    out.append(batch[k])
                for k in range(half, self.batch_size, 2):
                    if k + 1 < self.batch_size:
                        out.append((batch[k], batch[k + 1], 0))
                        out.append((batch[k], batch[k + 1], 1))
                yield out
                batch = []
        if len(batch) > 0 and not self.drop_last:
            while len(batch) < self.batch_size:
                batch.append(batch[-1])
            out = []
            for k in range(half):
                out.append(batch[k])
            for k in range(half, self.batch_size, 2):
                if k + 1 < self.batch_size:
                    out.append((batch[k], batch[k + 1], 0))
                    out.append((batch[k], batch[k + 1], 1))
            yield out

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
