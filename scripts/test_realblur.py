import argparse
import logging
import os
from copy import deepcopy
from os import path as osp

import numpy as np
import torch
from tqdm import tqdm

from basicsr.data import create_dataloader, create_dataset
from basicsr.models.archs import define_network
from basicsr.utils import compute_realblur_metrics, save_rgb_image
from basicsr.utils.options import parse as parse_options


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DFFormer on RealBlur.")
    parser.add_argument(
        "--yaml_file",
        required=True,
        type=str,
        help="Path to a training YAML file. Its val dataset section is used for evaluation.",
    )
    parser.add_argument("--weights", required=True, type=str, help="Checkpoint path.")
    parser.add_argument("--train_size", default=384, type=int, choices=[384, 512, -1], help="Inference patch size.")
    parser.add_argument("--lq_dir", default=None, type=str, help="Optional override for the blur directory.")
    parser.add_argument("--gt_dir", default=None, type=str, help="Optional override for the sharp directory.")
    parser.add_argument("--save_result", action="store_true", help="Save restored images.")
    parser.add_argument("--result_dir", default="./results/realblur_eval", type=str, help="Output directory.")
    parser.add_argument("--num_workers", default=4, type=int, help="Dataloader workers.")
    parser.add_argument("--device", default="cuda", type=str, help="Device, e.g. cuda, cuda:0, cpu.")
    return parser.parse_args()


def setup_logger():
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)


def load_state_dict(weights, device):
    checkpoint = torch.load(weights, map_location=device)
    if "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned_state_dict[key] = value
    return cleaned_state_dict


def to_numpy_image(tensor):
    return tensor.squeeze(0).detach().cpu().clamp_(0, 1).permute(1, 2, 0).numpy()

def main():
    args = parse_args()
    setup_logger()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    logging.info("Loading config from %s", args.yaml_file)
    opt = parse_options(args.yaml_file, is_train=False)
    opt["network_g"]["train_size"] = args.train_size if args.train_size > 0 else None

    dataset_opt = deepcopy(opt["datasets"]["val"])
    dataset_opt["phase"] = "val"
    dataset_opt["num_worker_per_gpu"] = args.num_workers
    dataset_opt.pop("crop_size", None)
    if args.lq_dir is not None:
        dataset_opt["dataroot_lq"] = osp.abspath(args.lq_dir)
    if args.gt_dir is not None:
        dataset_opt["dataroot_gt"] = osp.abspath(args.gt_dir)

    dataset = create_dataset(dataset_opt)
    dataloader = create_dataloader(dataset, dataset_opt, num_gpu=0, dist=False, sampler=None, seed=opt["manual_seed"])

    model = define_network(deepcopy(opt["network_g"]))
    model.load_state_dict(load_state_dict(args.weights, device), strict=True)
    model.to(device)
    model.eval()
    logging.info("Loaded weights from %s", args.weights)

    if args.save_result:
        os.makedirs(args.result_dir, exist_ok=True)

    psnr_values = []
    ssim_values = []
    blur_root = dataset.lq_folder

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            lq = batch["lq"].to(device)
            gt = batch["gt"]
            lq_path = batch["lq_path"][0]

            restored = model(lq)
            if isinstance(restored, list):
                restored = restored[-1]
            elif isinstance(restored, dict):
                restored = restored["img"]
                if isinstance(restored, list):
                    restored = restored[-1]

            pred_img = to_numpy_image(restored)
            gt_img = to_numpy_image(gt)

            psnr, ssim = compute_realblur_metrics(pred_img, gt_img)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

            if args.save_result:
                rel_path = osp.relpath(lq_path, blur_root)
                save_path = osp.join(args.result_dir, rel_path)
                save_rgb_image(save_path, pred_img)

    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))
    logging.info("Final PSNR: %.6f, Final SSIM: %.6f", avg_psnr, avg_ssim)


if __name__ == "__main__":
    main()
