"""
Inpaint image using mask from file (no SAM). Used for precomputing backgrounds
for DAM (Disentanglement Aggregation Model) in Re-ID.
Usage: load image and mask from disk, run LAMA inpainting, return/save result.
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img


def inpaint_image_with_mask_file(img_path, mask_path, lama_config, lama_ckpt, device="cuda", save_path=None):
    """
    Load image and mask from file, inpaint the mask region with LAMA, return (and optionally save) result.
    Mask file: e.g. 1500_c6s3_086567_01_mask.png for image 1500_c6s3_086567_01.jpg
    """
    img = load_img_to_array(img_path)
    mask = load_img_to_array(mask_path)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    if np.max(mask) > 1:
        mask = (mask > 127).astype(np.uint8) * 255
    else:
        mask = (mask > 0.5).astype(np.uint8) * 255
    img_inpainted = inpaint_img_with_lama(img, mask, lama_config, lama_ckpt, device=device)
    if save_path:
        save_array_to_img(img_inpainted, save_path)
    return img_inpainted


def setup_args(parser):
    parser.add_argument("--input_img", type=str, required=True, help="Path to input image")
    parser.add_argument("--input_mask", type=str, required=True, help="Path to mask image (e.g. xxx_mask.png)")
    parser.add_argument("--output_dir", type=str, default=None, help="If set, save inpainted image as xxx_bg.png here")
    parser.add_argument("--lama_config", type=str, default="./lama/configs/prediction/default.yaml")
    parser.add_argument("--lama_ckpt", type=str, required=True, help="Path to LAMA checkpoint dir (e.g. big-lama)")
    parser.add_argument("--device", type=str, default="cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = args.device if __import__("torch").cuda.is_available() else "cpu"
    save_path = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / (Path(args.input_img).stem + "_bg.png")
    inpaint_image_with_mask_file(
        args.input_img, args.input_mask,
        args.lama_config, args.lama_ckpt, device=device, save_path=save_path
    )
    if save_path:
        print("Saved:", save_path)
