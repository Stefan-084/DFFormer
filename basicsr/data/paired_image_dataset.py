import glob
import os
from os import path as osp

import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import paired_center_crop, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _scan_image_paths(root):
    paths = []
    for current_root, _, files in os.walk(root):
        for name in files:
            if osp.splitext(name)[1].lower() in IMG_EXTENSIONS:
                full_path = osp.join(current_root, name)
                rel_path = osp.relpath(full_path, root)
                paths.append(rel_path.replace("\\", "/"))
    return sorted(paths)


def _find_matching_gt(gt_root, rel_path):
    candidate = osp.join(gt_root, rel_path)
    if osp.exists(candidate):
        return candidate

    stem = osp.splitext(osp.join(gt_root, rel_path))[0]
    matches = glob.glob(stem + ".*")
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot find GT image for relative path: {rel_path}")


class Dataset_PairedImage(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = dict(opt["io_backend"])
        self.mean = opt.get("mean")
        self.std = opt.get("std")
        self.gt_folder = opt["dataroot_gt"]
        self.lq_folder = opt["dataroot_lq"]
        self.geometric_augs = opt.get("geometric_augs", False)

        if self.gt_folder is None or self.lq_folder is None:
            raise ValueError("dataroot_gt and dataroot_lq must be set for Dataset_PairedImage.")

        lq_rel_paths = _scan_image_paths(self.lq_folder)
        if not lq_rel_paths:
            raise FileNotFoundError(f"No images found under {self.lq_folder}")

        self.paths = []
        for rel_path in lq_rel_paths:
            self.paths.append(
                {
                    "lq_path": osp.join(self.lq_folder, rel_path),
                    "gt_path": _find_matching_gt(self.gt_folder, rel_path),
                }
            )

    def __getitem__(self, index):
        if self.file_client is None:
            backend_type = self.io_backend_opt.pop("type")
            self.file_client = FileClient(backend_type, **self.io_backend_opt)

        scale = self.opt["scale"]
        index = index % len(self.paths)

        gt_path = self.paths[index]["gt_path"]
        img_gt = imfrombytes(self.file_client.get(gt_path, "gt"), float32=True)

        lq_path = self.paths[index]["lq_path"]
        img_lq = imfrombytes(self.file_client.get(lq_path, "lq"), float32=True)

        if self.opt["phase"] == "train":
            gt_size = self.opt.get("gt_size", 0)
            if gt_size:
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
        elif self.opt["phase"] in ("val", "test"):
            crop_size = self.opt.get("crop_size")
            if crop_size:
                img_gt, img_lq = padding(img_gt, img_lq, crop_size)
                img_gt, img_lq = paired_center_crop(img_gt, img_lq, crop_size, scale, gt_path)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {"lq": img_lq, "gt": img_gt, "lq_path": lq_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.paths)
