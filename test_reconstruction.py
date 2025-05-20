#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

# Import the mock UniK3D model
from UniK3D.unik3d.models import UniK3D

def depth_to_points(depth, intrinsic=None):
    """Convert depth map to points in 3D space"""
    if intrinsic is None:
        # Default intrinsic matrix (approximation)
        h, w = depth.shape
        fx = fy = max(h, w) * 0.8
        cx, cy = w / 2, h / 2
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    
    # Invert intrinsic matrix
    Kinv = np.linalg.inv(intrinsic)
    
    # Create mesh grid of pixel coordinates
    h, w = depth.shape
    x = np.arange(w)
    y = np.arange(h)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # Add z=1
    coord = coord.astype(np.float32)
    
    # Reshape for matrix multiplication
    depth_expanded = depth[:, :, np.newaxis, np.newaxis]
    coord_expanded = coord[:, :, :, np.newaxis]
    
    # Calculate 3D points
    pts3D = depth_expanded * (Kinv @ coord_expanded)
    pts3D = pts3D[:, :, :3, 0]
    
    return pts3D

def create_point_cloud(pts3D, colors=None):
    """Create Open3D point cloud from 3D points and colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3D.reshape(-1, 3))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3) / 255.0)
    return pcd

def create_object_pcd(image_points, image_rgb, mask):
    """Create object point cloud using mask"""
    points = image_points[mask]
    colors = image_rgb[mask] / 255.0

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def colorize_depth(depth_map):
    """Convert depth map to colorized visualization"""
    normalized_depth = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
    colorized = plt.cm.viridis(normalized_depth)
    colorized = (colorized[:, :, :3] * 255).astype(np.uint8)
    return colorized

def main():
    # 1. Load an image
    try:
        image_path = "demo_images/urban.png"
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Create output directory
        output_dir = "./reconstruction_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Generate depth map using mock UniK3D
        print("Generating depth map using mock UniK3D...")
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        
        unik3d = UniK3D()
        unik3d_result = unik3d.infer(image_tensor)
        depth_map = unik3d_result["depth"]
        
        # Save depth visualization
        depth_vis = colorize_depth(depth_map)
        cv2.imwrite(os.path.join(output_dir, "depth_vis.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))
        
        # 3. Convert depth to points
        print("Converting depth to 3D points...")
        pts3D = depth_to_points(depth_map)
        
        # 4. Create point cloud
        print("Creating point cloud...")
        pcd = create_point_cloud(pts3D, image_rgb)
        
        # 5. Create a fake object mask (center rectangle)
        mask_h, mask_w = h // 2, w // 2
        mask = np.zeros((h, w), dtype=bool)
        mask[h//4:h//4+mask_h, w//4:w//4+mask_w] = True
        
        # 6. Create object point cloud
        print("Creating object point cloud with mask...")
        object_pcd = create_object_pcd(pts3D, image_rgb, mask)
        
        # 7. Save point clouds
        print("Saving point clouds...")
        o3d.io.write_point_cloud(os.path.join(output_dir, "full_pcd.ply"), pcd)
        o3d.io.write_point_cloud(os.path.join(output_dir, "object_pcd.ply"), object_pcd)
        
        # 8. Visualize results (optional)
        print("Visualization saved.")
        print(f"Full point cloud points: {len(pcd.points)}")
        print(f"Object point cloud points: {len(object_pcd.points)}")
        
        print(f"Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 