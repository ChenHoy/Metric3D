#! /usr/bin/env python3

import argparse
from pathlib import Path
import random
import time
import os
import ipdb
from typing import Tuple, List, Union, Dict
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import cv2
from tqdm.auto import tqdm


import matplotlib
import torch
import open3d as o3d


"""
Single image inference with MetricDepth3D
- Predict depth from a single image and view the result both in 2D and 3D.
- Run on folder of images and save the depth maps in output folder.
"""


EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
# EXTENSION_LIST = [".jpg", ".jpeg"]  # Use this on Replica

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def plot_2d(rgb: Image, depth: np.ndarray, cmap: str) -> None:
    import matplotlib.pyplot as plt

    # Colorize manually to appealing colormap
    percentile = 0.03
    min_depth_pct = np.percentile(depth, percentile)
    max_depth_pct = np.percentile(depth, 100 - percentile)
    depth_colored = colorize_depth_maps(
        depth, min_depth_pct, max_depth_pct, cmap=cmap
    ).squeeze()  # [3, H, W], value in (0, 1)
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # Plot the Image, Depth, and Uncertainty side-by-side in a 1x2 grid
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb)
    ax[0].set_title("Image")
    ax[1].imshow(chw2hwc(depth_colored))
    ax[1].set_title("Depth")
    ax[0].axis("off"), ax[1].axis("off")

    plt.show()


def get_calib_heuristic(ht: int, wd: int, heuristic: str = "generic") -> np.ndarray:
    """On in-the-wild data we dont have any calibration file.
    Since we optimize this calibration as well, we can start with an initial guess
    using the heuristic from DeepV2D and other papers"""
    cx, cy = wd // 2, ht // 2
    if heuristic == "teed":
        fx, fy = wd * 1.2, wd * 1.2
    else:
        fx, fy = (ht + wd) / 2.0, (ht + wd) / 2.0

    return fx, fy, cx, cy


def plot_2d_with_uncertainty(rgb: Image, depth: np.ndarray, uncertainty: np.ndarray, cmap: str) -> None:
    import matplotlib.pyplot as plt

    # Colorize manually to appealing colormap
    percentile = 0.03
    min_depth_pct = np.percentile(depth, percentile)
    max_depth_pct = np.percentile(depth, 100 - percentile)
    depth_colored = colorize_depth_maps(
        depth, min_depth_pct, max_depth_pct, cmap=cmap
    ).squeeze()  # [3, H, W], value in (0, 1)
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # Plot the Image, Depth, and Uncertainty side-by-side in a 1x3 grid
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(rgb)
    ax[0].set_title("Image")
    ax[1].imshow(chw2hwc(depth_colored))
    ax[1].set_title("Depth")
    ax[2].imshow(uncertainty)
    ax[2].set_title("Uncertainty")
    ax[0].axis("off"), ax[1].axis("off"), ax[2].axis("off")

    plt.show()


def plot_3d(rgb: Image, depth: np.ndarray):
    """Use Open3d to plot the 3D point cloud from the monocular depth and input image."""

    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    invalid = filter_prediction(depth).flatten()

    # Get 3D point cloud from depth map
    depth = depth.squeeze()
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.flatten()
    y = y.flatten()
    depth = depth.flatten()

    # Convert to 3D points
    fx, fy, cx, cy = get_calib_heuristic(h, w)
    # Unproject
    x3 = (x - cx) * depth / fx
    y3 = (y - cy) * depth / fy
    z3 = depth

    # Convert to Open3D format
    xyz = np.stack([x3, y3, z3], axis=1)
    rgb = np.stack([rgb[:, :, 0].flatten(), rgb[:, :, 1].flatten(), rgb[:, :, 2].flatten()], axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    # Plot the point cloud
    o3d.visualization.draw_geometries([pcd])


def filter_prediction(depth_pred: np.ndarray, percentile: float = 0.01) -> np.ndarray:
    """Filter out invalid outlier depths according the distribution.
    Most of outliers are likely at near 0 and very far away compared to the actual object depths.
    """
    min_depth_pct = np.percentile(depth_pred, percentile)
    max_depth_pct = np.percentile(depth_pred, 100 - percentile)
    # Return mask for invalid predictions
    return np.logical_or(depth_pred < min_depth_pct, depth_pred > max_depth_pct)


def seed_all(seed: int = 0):
    """
    Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # Random seed
    seed_all(1234)

    # Device
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    print(f"device = {device}")

    # -------------------- Model --------------------
    model_name = "metric3d_" + "vit_giant2"
    model = torch.hub.load("yvanyin/metric3d", model_name, pretrain=True)
    model.cuda().eval()

    # -------------------- Camera model --------------------
    # rgb_file = "/media/data/nerf/ours/kitchen/raw/images/frame_01328.png"
    rgb_file = "/media/data/KITTI/raw/sequences/21/image_2/002710.png"
    intrinsic = [718.856, 718.856, 607.1928, 185.2157]  # Kitti intrinsics

    if "vit" in model_name:
        input_size = (616, 1064)  # for vit model
    else:
        input_size = (544, 1216)  # for convnext model

    # -------------------- Load image & Preprocess --------------------
    rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic, hold depth
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    # normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()

    # -------------------- Inference --------------------
    with torch.no_grad():
        pred_depth, confidence, output_dict = model.inference({"input": rgb})

    # -------------------- Post-processing  --------------------
    # unpad
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]
    ]
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None, :, :], rgb_origin.shape[:2], mode="bilinear"
    ).squeeze()
    ###################### canonical camera space ######################
    # de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0  # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)

    # FIXME is the metric depth now accurately computed?
    # we somehow dont get the metric depth here :/
    ipdb.set_trace()


if __name__ == "__main__":
    main()
