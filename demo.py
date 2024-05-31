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


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image depth estimation using Zoedepth.")
    parser.add_argument(
        "--model",
        type=str,
        default="vit_giant2",
        choices=["convnext_large", "vit_small", "vit_large", "vit_giant2"],
        help="""Checkpoint path or hub name. The bigger the better here""",
    )

    parser.add_argument(
        "-i",
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    # depth map colormap
    parser.add_argument(
        "--depth_cmap",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
    # TODO how to supply these correctly for our datasets?
    # TODO chen: write a new file for using dataloaders and demo uses the in-the-wild heuristics
    parser.add_argument("--intrinsics", type=str, default=None, help="Path to camera intrinsics.")
    parser.add_argument("-b", "--batch-size", type=int, default=None, help="Batch size")

    return parser.parse_args()


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


def postprocess_outputs(
    depth: torch.Tensor,
    conf: torch.Tensor,
    normals: torch.Tensor,
    pad_info: list,
    original_shape: Tuple[int, int],
    scale_info: float,
    normalize_scale: float,
) -> torch.Tensor:
    """Resize und unpad depth map to original size and scale to metric depth."""
    # Unpad
    depth = depth[pad_info[0] : depth.shape[0] - pad_info[1], pad_info[2] : depth.shape[1] - pad_info[3]]
    conf = conf[pad_info[0] : conf.shape[0] - pad_info[1], pad_info[2] : conf.shape[1] - pad_info[3]]
    normals = normals[:, pad_info[0] : normals.shape[-2] - pad_info[1], pad_info[2] : normals.shape[-1] - pad_info[3]]
    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(depth[None, None, :, :], original_shape, mode="bilinear").squeeze()
    conf = torch.nn.functional.interpolate(conf[None, None, :, :], original_shape, mode="bilinear").squeeze()
    normals = torch.nn.functional.interpolate(normals[None, ...], original_shape, mode="bilinear").squeeze()
    # Scale to metric depth
    pred_depth = pred_depth * normalize_scale / scale_info
    pred_depth = torch.clamp(pred_depth, 0, 300)
    return pred_depth, conf, normals


def build_camera_model(H: int, W: int, intrinsics: Union[List, Tuple, np.ndarray]) -> np.ndarray:
    """
    Encode the camera intrinsic parameters (focal length and principle point) to a 4-channel map.
    """
    fx, fy, u0, v0 = intrinsics
    f = (fx + fy) / 2.0
    # principle point location
    x_row = np.arange(0, W).astype(np.float32)
    x_row_center_norm = (x_row - u0) / W
    x_center = np.tile(x_row_center_norm, (H, 1))  # [H, W]

    y_col = np.arange(0, H).astype(np.float32)
    y_col_center_norm = (y_col - v0) / H
    y_center = np.tile(y_col_center_norm, (W, 1)).T  # [H, W]

    # FoV
    fov_x = np.arctan(x_center / (f / W))
    fov_y = np.arctan(y_center / (f / H))

    cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
    return cam_model


def resize_for_input(
    image: np.ndarray,
    output_shape: Tuple[int, int],
    intrinsics: np.ndarray,
    canonical_shape,
    to_canonical_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, List[int], float]:
    """
    Resize the input according to https://github.com/YvanYin/Metric3D/blob/main/mono/utils/do_test.py
    Resizing consists of two processed, i.e. 1) to the canonical space (adjust the camera model); 2) resize the image while the camera model holds. Thus the
    label will be scaled with the resize factor.
    """
    ### Prepare the image to right tensor
    padding = [123.675, 116.28, 103.53]
    h, w, _ = image.shape
    resize_ratio_h = output_shape[0] / canonical_shape[0]
    resize_ratio_w = output_shape[1] / canonical_shape[1]
    to_scale_ratio = min(resize_ratio_h, resize_ratio_w)
    resize_ratio = to_canonical_ratio * to_scale_ratio

    reshape_h = int(resize_ratio * h)
    reshape_w = int(resize_ratio * w)

    pad_h = max(output_shape[0] - reshape_h, 0)
    pad_w = max(output_shape[1] - reshape_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)

    # Resize
    image = cv2.resize(image, dsize=(reshape_w, reshape_h), interpolation=cv2.INTER_LINEAR)
    # Pad
    image = cv2.copyMakeBorder(
        image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding
    )

    ### Get camera model
    # Resize, adjust principle point
    # NOTE we already scale the focal length in test_data_scalecanuo()
    intrinsics[2] = intrinsics[2] * to_scale_ratio
    intrinsics[3] = intrinsics[3] * to_scale_ratio
    cam_model = build_camera_model(reshape_h, reshape_w, intrinsics)
    cam_model = cv2.copyMakeBorder(
        cam_model, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=-1
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]
    label_scale_factor = 1 / to_scale_ratio

    return image, cam_model, pad_info, label_scale_factor


def transform_test_data_scalecano(
    rgb: np.ndarray, intrinsics: Union[List, np.ndarray, Tuple], data_basic: Dict, device: str = "cuda:0"
):
    """
    Pre-process the input for forwarding. Employ `label scale canonical transformation.'
        Args:
            rgb: input rgb image. [H, W, 3]
            intrinsic: camera intrinsic parameter, [fx, fy, u0, v0]
            data_basic: predefined canonical space in configs.
    """
    canonical_space = data_basic["canonical_space"]
    forward_size = data_basic["crop_size"]

    ori_h, ori_w, _ = rgb.shape
    # Average to single scale for focal length
    ori_focal = (intrinsics[0] + intrinsics[1]) / 2

    # Adjust focal length according to ratio
    # NOTE principal point is scaled in resize_for_input()
    canonical_focal = canonical_space["focal_length"]
    cano_label_scale_ratio = canonical_focal / ori_focal
    canonical_intrinsic = [
        intrinsics[0] * cano_label_scale_ratio,
        intrinsics[1] * cano_label_scale_ratio,
        intrinsics[2],
        intrinsics[3],
    ]

    # Resize & Pad
    rgb, cam_model, pad, resize_label_scale_ratio = resize_for_input(
        rgb, forward_size, canonical_intrinsic, [ori_h, ori_w], 1.0
    )

    # Normalize to ImageNet mean and std
    mean = torch.tensor([IMAGENET_MEAN]).float()[:, None]
    std = torch.tensor([IMAGENET_STD]).float()[:, None]
    rgb = torch.from_numpy(rgb).float() / 255.0  # Normalize [0, 255] -> [0, 1]
    rgb = torch.div((rgb - mean), std)
    rgb = rgb.permute(2, 0, 1).unsqueeze(0).to(device)

    # label scale factor
    label_scale_factor = cano_label_scale_ratio * resize_label_scale_ratio

    cam_model = torch.from_numpy(cam_model.transpose((2, 0, 1))).float()
    cam_model = cam_model[None, :, :, :].to(device)
    cam_model_stacks = [
        torch.nn.functional.interpolate(
            cam_model, size=(cam_model.shape[2] // i, cam_model.shape[3] // i), mode="bilinear", align_corners=False
        )
        for i in [2, 4, 8, 16, 32]
    ]
    return rgb, cam_model_stacks, pad, label_scale_factor


def barebone_preprocess(
    rgb_origin: np.ndarray, input_size: Tuple[int, int], intrinsics: np.ndarray
) -> Tuple[torch.Tensor, List[int], np.ndarray]:
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic, hold depth
    intrinsics = [intrinsics[0] * scale, intrinsics[1] * scale, intrinsics[2] * scale, intrinsics[3] * scale]
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
    return rgb, pad_info, intrinsics


def barebone_postprocess(
    pred_depth: torch.Tensor,
    confidence: torch.Tensor,
    normals: torch.Tensor,
    origin_shape: Tuple[int, int],
    pad_info: List[int],
    intrinsics: np.ndarray,
    max_depth: float = 300.0,
):
    # Unpad
    pred_depth = pred_depth[
        pad_info[0] : pred_depth.shape[-2] - pad_info[1], pad_info[2] : pred_depth.shape[-1] - pad_info[3]
    ]
    confidence = confidence[
        pad_info[0] : confidence.shape[-2] - pad_info[1], pad_info[2] : confidence.shape[-1] - pad_info[3]
    ]
    normals = normals[:, pad_info[0] : normals.shape[-2] - pad_info[1], pad_info[2] : normals.shape[-1] - pad_info[3]]

    # upsample to original size
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], origin_shape, mode="bilinear").squeeze()
    confidence = torch.nn.functional.interpolate(confidence[None, None, :, :], origin_shape, mode="bilinear").squeeze()
    normals = torch.nn.functional.interpolate(normals[None, ...], origin_shape, mode="bilinear").squeeze()

    ###################### canonical camera space ######################
    # de-canonical transform
    canonical_to_real_scale = intrinsics[0] / 1000.0  # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, max_depth)
    return pred_depth, confidence, normals


def main():
    args = parse_args()
    # Random seed
    if args.seed is None:
        seed = int(time.time())
    seed_all(seed)

    # Device
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    print(f"device = {device}")

    # -------------------- Data --------------------
    rgb_filename_list = glob(os.path.join(args.input_rgb_dir, "*"))
    rgb_filename_list = [f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST]
    rgb_filename_list = sorted(rgb_filename_list)
    n_images = len(rgb_filename_list)
    if n_images > 0:
        print(f"Found {n_images} images")
    else:
        raise RuntimeError(f"No image found in '{args.input_rgb_dir}'")

    # -------------------- Model --------------------
    model = torch.hub.load("yvanyin/metric3d", "metric3d_" + args.model, pretrain=True)
    model.cuda().eval()

    # -------------------- Camera model --------------------
    if args.intrinsics is not None:
        intrinsics_raw = np.loadtxt(args.intrinsics)
        intrinsics_raw = np.array([intrinsics_raw[:4]]).squeeze()
    else:
        ht, wd = cv2.imread(rgb_filename_list[0]).shape[:2]
        intrinsics_raw = get_calib_heuristic(ht, wd, heuristic="generic")

    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = 1

    if "vit" in args.model:
        input_size = (616, 1064)  # for vit model
    else:
        input_size = (544, 1216)  # for convnext model

    ### Canonical camera setting and basic data setting
    # This is according to the E300 camera (crop version)
    # canonical_space = dict(img_size=(540, 960), focal_length=1196.0)
    # depth_range = (0.9, 150)  # Depth range in canonical model
    # normalize_scale = depth_range[1]
    # data_basic = dict(
    #     canonical_space=canonical_space,
    #     depth_range=depth_range,
    #     depth_normalize=(0.006, 1.001),
    #     crop_size=input_size,  # NOTE this is somehow (512, 960) in their demo
    #     clip_depth_range=(0.9, 150),
    # )

    os.makedirs(args.output_dir, exist_ok=True)
    # -------------------- Inference and saving --------------------
    batch = {"input": [], "pads": [], "cam_model": [], "paths": [], "label_scale_factor": [], "orig_size": []}
    for i, rgb_path in tqdm(enumerate(rgb_filename_list), desc=f"Estimating depth", leave=True):

        ### Get preprocessed image
        rgb = cv2.imread(rgb_path)[:, :, ::-1]
        ht, wd = rgb.shape[:2]

        # NOTE chen: this is processing from some of their other scripts
        # rgb_input, cam_model_stacks, pad, label_scale_factor = transform_test_data_scalecano(
        #     rgb, intrinsics, data_basic, device=device
        # )
        # NOTE chen: this is processing from their huggingface script
        rgb_input, pad, intrinsics = barebone_preprocess(rgb, input_size, intrinsics_raw)

        # # Fill batch until reaching batch_size or if we reached the end of the dataset
        batch["input"].append(rgb_input[0])
        batch["orig_size"].append((ht, wd))
        batch["pads"].append(pad)
        batch["paths"].append(rgb_path)
        # batch["cam_model"].append(cam_model_stacks)
        # batch["label_scale_factor"].append(label_scale_factor)
        if len(batch["input"]) < batch_size and i < n_images - 1:
            continue

        ### Inference
        input_data = dict(input=torch.stack(batch["input"]), cam_model=None)
        pred_depth, confidence_out, output_dict = model.inference(input_data)

        ### Postprocess
        normal_out = output_dict["normal_out_list"][0]  # TODO why do we take the first element and not the last?

        for ii in range(len(pred_depth)):
            depth_pred, confidence, normals = barebone_postprocess(
                pred_depth[ii].squeeze(),
                confidence_out[ii].squeeze(),
                normal_out[ii].squeeze(),
                batch["orig_size"][ii],
                batch["pads"][ii],
                intrinsics,
            )

            # depth_pred, confidence, normals = postprocess_outputs(
            #     pred_depth[ii].squeeze(),
            #     confidence_out[ii].squeeze(),
            #     normal_out[ii].squeeze(),
            #     batch["pads"][ii],
            #     batch["orig_size"][ii],
            #     normalize_scale,
            #     batch["label_scale_factor"][ii],
            # )
            depth_pred, confidence, normals = (
                depth_pred.cpu().numpy(),
                confidence.cpu().numpy(),
                normals.cpu().numpy(),
            )
            # Normalize confidence, since this is [-1, 1]
            conf_range = confidence.max() - confidence.min()
            confidence = (confidence - confidence.min()) / conf_range

            # TODO visualize normals if wanted
            # plot_2d(rgb, depth_pred, cmap=args.depth_cmap)
            # plot_2d_with_uncertainty(rgb, depth_pred, confidence, cmap=args.depth_cmap)
            # plot_3d(rgb, depth_pred)

            # Save depth map with numpy
            fname = Path(batch["paths"][ii]).stem + ".npy"
            output_path = os.path.join(args.output_dir, fname)
            np.save(output_path, depth_pred)

        # Reset batch
        batch = {"input": [], "pads": [], "cam_model": [], "paths": [], "label_scale_factor": [], "orig_size": []}


if __name__ == "__main__":
    main()
