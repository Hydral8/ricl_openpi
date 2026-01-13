# RICL Integration Guide

This document describes Steps 2 & 3 of integrating the LeRobot retriever with RICL.

## Step 2: Retrieval Contract Definition

### Field Mapping

| LeRobot Dataset Field | RICL Expected Field | Notes |
|----------------------|---------------------|-------|
| `observation.state` | `retrieved_{i}_state` | Direct mapping |
| `observation.images.image` or `observation.images.{camera}` | `retrieved_{i}_top_image` | Heuristic mapping based on camera name keywords |
| `observation.images.{camera}` | `retrieved_{i}_wrist_image` | Heuristic mapping |
| `observation.images.{camera}` | `retrieved_{i}_right_image` | Heuristic mapping |
| `action` | `retrieved_{i}_actions` | Chunk of length `action_horizon`, padded if needed |
| `dataset.meta.tasks.index[task_index]` | `retrieved_{i}_prompt` | Task description string |

### Camera Name Mapping Heuristics

The `map_camera_to_ricl_keys()` function uses keyword matching:

- **top_image**: Matches keywords `["top", "base", "head", "cam_high", "image"]`
- **wrist_image**: Matches keywords `["wrist", "hand", "cam_left_wrist", "cam_right_wrist"]`
- **right_image**: Matches keywords `["right", "cam_right"]`

If a camera isn't found, falls back to:
1. Using other available cameras
2. Duplicating `top_image` if needed

### Distance Computation

RICL expects `exp_lamda_distances` of shape `(knn_k + 1, 1)`:
- First value: 0.0 (distance from first retrieved demo to itself)
- Next `knn_k - 1` values: Distances from first retrieved demo to other retrieved demos
- Last value: Distance from query to first retrieved demo

Formula: `exp(-λ * normalized_distance)` where distances are clipped to `[0, max_dist]` and normalized.

## Step 3: Implementation

### Main Function: `lerobot_retrieve_for_ricl()`

Located in `openpi.policies.lerobot_retrieval_bridge`, this function:

1. **Takes query observation** with `query_top_image` (or similar)
2. **Performs retrieval** using either:
   - Two-stage: `retrieve_rar()` with text + image queries
   - Single-stage: `image_retrieval()` with image-only query
3. **Extracts single timestep** from each retrieved segment (using `start_idx` as the best match)
4. **Maps fields** from LeRobot format to RICL format
5. **Computes distances** and `exp_lamda_distances`
6. **Returns** observation dict with all `retrieved_{i}_*` keys

### Helper Functions

- `get_action_chunk_at_inference_time()`: Extracts action chunk with padding
- `extract_single_timestep_from_segment()`: Extracts single timestep from segment
- `map_camera_to_ricl_keys()`: Maps camera names to RICL keys

### Usage Example (Direct)

```python
from rar_client.base import init_lerobot_retriever
from openpi.policies.lerobot_retrieval_bridge import lerobot_retrieve_for_ricl

# Initialize retriever
handle = init_lerobot_retriever(
    dataset_dir="./libero_10_dataset",
    text_index_path="./text_index.faiss",
    image_index_path="./image_index.faiss",
)

# Query observation
query_obs = {
    "query_top_image": query_image_array,  # numpy array or PIL Image
    "query_state": query_state_array,
}

# Retrieve and format for RICL
ricl_obs = lerobot_retrieve_for_ricl(
    query_obs=query_obs,
    handle=handle,
    knn_k=5,  # num_retrieved_observations
    action_horizon=32,
    lamda=10.0,
    max_dist=1.0,  # or load from assets/max_distance.json
    use_text_retrieval=False,  # or True with text_query
)

# ricl_obs now contains:
# - retrieved_0_state, retrieved_0_top_image, retrieved_0_wrist_image, ...
# - retrieved_1_state, retrieved_1_top_image, ...
# - exp_lamda_distances
# - All original query_obs keys
```

## Step 4: Integration into RiclPolicy ✅

**COMPLETED**: The LeRobot retriever is now integrated into `RiclPolicy` with backward compatibility.

### Usage via RiclPolicy

```python
from openpi.policies import RiclPolicy
from openpi.models import pi0_fast_ricl

# Create model
model = pi0_fast_ricl.Pi0FASTRicl(config, rngs=...)

# Option 1: Use LeRobot retrieval (NEW)
policy = RiclPolicy(
    model,
    use_lerobot_retrieval=True,
    lerobot_dataset_dir="./libero_10_dataset",
    lerobot_text_index_path="./text_index.faiss",
    lerobot_image_index_path="./image_index.faiss",
    use_action_interpolation=True,
    lamda=10.0,
    action_horizon=32,
    # ... other args
)

# Option 2: Use old .npz retrieval (backward compatible)
policy = RiclPolicy(
    model,
    demos_dir="./preprocessing/collected_demos/...",
    use_action_interpolation=True,
    lamda=10.0,
    action_horizon=32,
    # ... other args
)

# Both work the same way:
obs = {"query_top_image": ..., "query_state": ...}
result = policy.infer(obs)
```

### Implementation Details

- **Backward Compatible**: If `use_lerobot_retrieval=False` (default), uses the old `.npz` method
- **New Parameters**: 
  - `use_lerobot_retrieval`: Enable LeRobot retrieval
  - `lerobot_dataset_dir`: Path to LeRobot dataset
  - `lerobot_text_index_path`: Path to text FAISS index
  - `lerobot_image_index_path`: Path to image FAISS index
- **Automatic**: The `retrieve()` method automatically uses the correct retrieval backend based on initialization

### Requirements

- Install the `rar_client` SDK package (from `/path/to/sdk`)
- Ensure FAISS indices are built for your dataset
- Ensure LeRobot dataset is properly formatted
