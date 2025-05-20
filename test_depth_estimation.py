#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from mmengine import Config

# Import our scene graph generator
from dataset_pipeline.generate_3dsg import GeneralizedSceneGraphGenerator

def colorize_depth(depth_map, min_depth=None, max_depth=None):
    """Convert depth map to colorized visualization"""
    if min_depth is None:
        min_depth = np.min(depth_map)
    if max_depth is None:
        max_depth = np.max(depth_map)
    
    # Normalize to 0-1
    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth + 1e-8)
    # Apply colormap
    colorized = plt.cm.viridis(normalized_depth)
    # Convert to uint8
    colorized = (colorized[:, :, :3] * 255).astype(np.uint8)
    return colorized

def main():
    parser = argparse.ArgumentParser(description="Test depth estimation with UniK3D and Metric3D_v2")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--config", type=str, default="configs/v2_hf_llm.py", help="Path to config file")
    parser.add_argument("--use_unik3d", action="store_true", help="Try to use UniK3D (will fallback to Metric3D if not available)")
    parser.add_argument("--output_dir", type=str, default="./depth_outputs", help="Directory to save output")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the config
    cfg = Config.fromfile(args.config)
    
    # Override config based on arguments
    cfg.use_unik3d = args.use_unik3d
    
    # Initialize the scene graph generator
    print(f"Initializing generator with config: {args.config}")
    print(f"UniK3D enabled: {cfg.use_unik3d}")
    
    try:
        generator = GeneralizedSceneGraphGenerator(
            config_path=args.config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Load the image
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image not found at {args.image}")
        
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            raise ValueError(f"Could not read image from {args.image}")
        
        # Resize image
        h, w = image_bgr.shape[:2]
        target_h = cfg.get("image_resize_height", 640)
        scale = target_h / h
        target_w = int(w * scale)
        image_bgr_resized = cv2.resize(image_bgr, (target_w, target_h))
        
        # Convert to tensor for depth estimation
        image_tensor = torch.from_numpy(image_bgr_resized).float()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC to CHW
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Get depth
        print("Estimating depth...")
        depth = generator._get_depth(image_tensor)
        print(f"Depth estimation complete. Shape: {depth.shape}")
        
        # Save depth visualization
        colorized_depth = colorize_depth(depth)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_prefix = f"{base_name}_unik3d" if cfg.use_unik3d else f"{base_name}_metric3d"
        
        # Save original image
        original_rgb = cv2.cvtColor(image_bgr_resized, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(args.output_dir, f"{output_prefix}_input.jpg"), image_bgr_resized)
        
        # Save depth visualization
        cv2.imwrite(os.path.join(args.output_dir, f"{output_prefix}_depth.jpg"), colorized_depth)
        
        # Save raw depth as numpy array
        np.save(os.path.join(args.output_dir, f"{output_prefix}_depth.npy"), depth)
        
        print(f"Results saved to {args.output_dir}")
        print(f"  Input image: {output_prefix}_input.jpg")
        print(f"  Depth visualization: {output_prefix}_depth.jpg")
        print(f"  Raw depth: {output_prefix}_depth.npy")
        
    except Exception as e:
        print(f"Error during depth estimation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 