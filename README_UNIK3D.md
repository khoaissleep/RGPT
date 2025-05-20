# UniK3D Integration for SpatialRGPT

This document explains how to use the UniK3D integration in SpatialRGPT, a 3D scene graph generation pipeline.

## Overview

UniK3D is a universal camera monocular 3D estimation model that can produce high-quality depth maps. It's integrated as an optional enhancement to the SpatialRGPT pipeline, with automatic fallback to Metric3D_v2 when UniK3D is not available.

## Requirements

To use UniK3D, you need:
- Python 3.11+ (UniK3D requirement)
- PyTorch 2.0+
- CUDA capable GPU (recommended)
- UniK3D package installed

If these requirements are not met, the system will automatically fall back to using Metric3D_v2 for depth estimation.

## Installation

There are two options for installation:

### Option 1: With UniK3D (Recommended for best depth estimation)

```bash
# Install SpatialRGPT dependencies
pip install -r requirements.txt

# Install UniK3D from source
git clone https://github.com/lpiccinelli-eth/UniK3D.git
cd UniK3D
pip install -e .
cd ..
```

### Option 2: Without UniK3D (Fallback mode)

```bash
# Install SpatialRGPT dependencies only
pip install -r requirements.txt
```

## Configuration

The configuration for depth estimation is in the config files. Key parameters:

```python
# Enable/disable UniK3D (set to False to always use Metric3D_v2)
use_unik3d = True  

# UniK3D resolution level
unik3d_resolution_level = 9

# Depth fusion parameters
depth_fusion = dict(
    method = "weighted_average",  # Options: "weighted_average" or "selective"
    weights = [0.5, 0.5],  # Weights for Metric3D and UniK3D respectively
    confidence_threshold = 0.7  # Threshold for selective fusion
)
```

## Usage

### Using the Scene Graph Generator

```python
from dataset_pipeline.generate_3dsg import GeneralizedSceneGraphGenerator

# Initialize with UniK3D support
generator = GeneralizedSceneGraphGenerator(
    config_path="configs/v2_hf_llm.py",
    device="cuda"
)

# Generate 3D scene graph
detections, facts, rephrased_qas = generator.generate_facts(
    "path/to/image.jpg",
    run_llm_rephrase=True
)
```

### Testing Depth Estimation

Use the provided test script to compare depth estimation:

```bash
# Test with UniK3D (will fallback to Metric3D if not available)
python test_depth_estimation.py --image path/to/image.jpg --use_unik3d

# Test with Metric3D_v2 only
python test_depth_estimation.py --image path/to/image.jpg
```

The script will save:
- Input image
- Depth visualization
- Raw depth data (as numpy array)

## How the Fallback Mechanism Works

The pipeline implements a smart fallback mechanism:

1. The system first checks if UniK3D is available and enabled:
   ```python
   use_unik3d = self.cfg.get("use_unik3d", False) and UNIK3D_AVAILABLE
   ```

2. If UniK3D is available, it tries to use it:
   ```python
   if use_unik3d:
       # Try to use UniK3D...
   ```

3. If any error occurs during UniK3D initialization or inference, it logs a warning and falls back to Metric3D_v2:
   ```python
   except Exception as e:
       self.logger.warning(f"Error using UniK3D: {e}. Falling back to Metric3D_v2.")
   ```

4. If UniK3D is not available, it directly uses Metric3D_v2:
   ```python
   # Fall back to Metric3D_v2
   self.logger.info("Using Metric3D_v2 for depth estimation")
   depth = self.reconstructor.depth_model.infer_depth(image_tensor)
   ```

This ensures robustness in various environments without crashing the pipeline when UniK3D is unavailable.

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'unik3d'**
   - This means UniK3D is not installed. The system will automatically use Metric3D_v2.

2. **CUDA out of memory**
   - UniK3D can be memory-intensive. Try reducing the image resolution or using a smaller model variant.

3. **Depth map quality issues**
   - Try adjusting the fusion weights in the config to give more weight to the better-performing model.

### Getting Help

For further assistance, please open an issue in the repository or contact the maintainers.

## License

This integration is subject to the licenses of both SpatialRGPT and UniK3D. Please refer to their respective license files. 