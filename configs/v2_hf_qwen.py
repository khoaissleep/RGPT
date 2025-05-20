# UniK3D Configuration
use_unik3d = True  # Enable/disable UniK3D integration
depth_fusion = dict(
    method = "weighted_average",  # Options: "weighted_average" or "selective"
    weights = [0.5, 0.5],  # Weights for Metric3D and UniK3D respectively
    confidence_threshold = 0.7  # Threshold for selective fusion
) 