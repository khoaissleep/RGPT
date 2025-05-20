# SceneGraph Generator: LLM + Vision Configuration

# Vision-related configs
# SAM-related configurations
sam_checkpoint = "pretrained/sam_models/sam_vit_h_4b8939.pth"
sam_model_type = "vit_h"

# Grounding DINO configs
grounding_dino_config = "pretrained/groundingdino/GroundingDINO_SwinT_OGC.py"
grounding_dino_checkpoint = "pretrained/groundingdino/groundingdino_swint_ogc.pth"
box_threshold = 0.35
text_threshold = 0.25
nms_threshold = 0.5

# Perspective Fields configs
perspective_model_variant = "geo_calib"  # or "perspective_fields"

# Tag2Text settings
tag_threshold = 0.68
specified_tags = ["furniture", "electronics", "vehicle", "food", "plant", "animal", "accessory", "sports", "decoration"]
class_set = None  # Use Tag2Text by default
remove_classes = ["ground", "floor", "wall", "building", "sky", "ceiling", "road", "background"]

# Scene settings
image_resize_height = 640  # Resize input images to this height
log_dir = "./temp_outputs/log"
wis3d_folder = None  # Will be auto-generated if not provided
vis = False  # Enable/disable visualization

# PointCloud reconstruction configs
min_points_threshold = 100
min_points_threshold_after_denoise = 50
downsample_voxel_size = 0.01
dbscan_eps = 0.03
dbscan_min_points = 10

# Mask processing thresholds
mask_contain_th1 = 0.05
mask_contain_th2 = 0.05

# Depth estimation with UniK3D integration
use_unik3d = True  # Enable/disable UniK3D integration
unik3d_resolution_level = 9  # Resolution level for UniK3D depth estimation
use_camera_params = False  # Use camera parameters for improved depth estimation if available
depth_fusion = dict(
    method = "weighted_average",  # Options: "weighted_average" or "selective"
    weights = [0.2, 0.8],  # Weights for Metric3D and UniK3D respectively (ưu tiên UniK3D hơn)
    confidence_threshold = 0.7  # Threshold for confidence in selective fusion
)

# Language model configurations - DEFAULT TO QWEN2
# Change to your preferred model (Llama3, Qwen2, etc.)
llm_model_name_hf = "Qwen/Qwen2-7B-Instruct"  # Hugging Face model name
llm_max_new_tokens = 512  # Maximum number of tokens to generate
llm_max_retries = 3  # Number of attempts for LLM generation
llm_temperature = 0.1  # Temperature for generation - keep low for factual responses

# Fact template generation settings
global_qs_list = [
    "What is the spatial relationship between {det_i} and {det_j}?",
    "Does {det_i} support {det_j}?",
    "Is {det_i} above {det_j}?",
    "Is {det_i} behind {det_j}?",
    "Is {det_i} left of {det_j}?",
    "What is the distance between {det_i} and {det_j}?"
] 