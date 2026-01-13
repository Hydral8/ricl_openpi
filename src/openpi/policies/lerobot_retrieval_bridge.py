"""Bridge module to convert retrieve_rar output to RICL-compatible format.

This module implements Steps 2 & 3:
- Step 2: Defines the mapping contract from LeRobot dataset fields to RICL expected fields
- Step 3: Implements lerobot_retrieve_for_ricl() wrapper function

Field Mapping (Step 2):
=======================
LeRobot Dataset Fields          ->  RICL Expected Fields
--------------------------------     ---------------------------
observation.state               ->  retrieved_{i}_state
observation.images.image        ->  retrieved_{i}_top_image
observation.images.wrist_image  ->  retrieved_{i}_wrist_image
(N/A - dataset doesn't have)    ->  retrieved_{i}_right_image (duplicated from wrist_image)
action                          ->  retrieved_{i}_actions (chunk of length action_horizon)
dataset.meta.tasks.index[task]  ->  retrieved_{i}_prompt

Note: This dataset has:
- observation.images.image (main camera, 256x256x3)
- observation.images.wrist_image (wrist camera, 256x256x3)
- No right_image camera, so we duplicate wrist_image for RICL compatibility

RICL also expects:
- exp_lamda_distances: shape (knn_k + 1, 1) with exp(-λ * normalized_distance)
"""

from typing import Any

import numpy as np

try:
    from rar_client.base import RetrieverHandle, image_retrieval, text_retrieval, get_task_episode_indices, retrieve_rar
    from rar_client.models import get_dino_image_embedding
except ImportError:
    raise ImportError(
        "rar_client package not found. Install the SDK package: "
        "pip install -e /path/to/sdk or add it to your PYTHONPATH"
    )


def get_action_chunk_at_inference_time(actions: np.ndarray, step_idx: int, action_horizon: int) -> np.ndarray:
    """Extract action chunk starting from step_idx, padding if needed.
    
    This mirrors RICL's get_action_chunk_at_inference_time logic.
    If we run out of actions, pads with zeros for joint velocities and last gripper position.
    """
    num_steps = len(actions)
    action_chunk = []
    for i in range(action_horizon):
        if step_idx + i < num_steps:
            action_chunk.append(actions[step_idx + i])
        else:
            # Pad: zeros for all but last dim (gripper), use last gripper value
            if actions.shape[-1] > 1:
                padded = np.concatenate([
                    np.zeros(actions.shape[-1] - 1, dtype=np.float32),
                    actions[-1, -1:]
                ], axis=0)
            else:
                padded = actions[-1]
            action_chunk.append(padded)
    return np.stack(action_chunk, axis=0)


def extract_single_timestep_from_segment(
    handle: RetrieverHandle,
    segment: dict[str, Any],
    timestep_offset: int = 0,
) -> dict[str, Any]:
    """Extract a single timestep from a retrieved segment.
    
    Args:
        handle: RetrieverHandle with dataset access
        segment: Segment dict from image_retrieval with keys:
            - start_idx, end_idx: frame range
            - data: dict with state, action, episode_index, task_index
            - frames_data: dict with images (if fetch_frames=True)
        timestep_offset: Offset from start_idx to pick the timestep (0 = use start_idx)
    
    Returns:
        dict with keys: state, action, images, episode_index, task_index, frame_idx
    """
    frame_idx = segment["start_idx"] + timestep_offset
    # Clamp to valid range
    frame_idx = min(frame_idx, segment["end_idx"])
    
    # Extract from segment data (already sliced from rest_ds)
    segment_data = segment["data"]
    local_idx = frame_idx - segment["start_idx"]
    
    # Handle dict-of-arrays format from HF dataset
    if isinstance(segment_data, dict):
        state = segment_data["observation.state"][local_idx]
        action = segment_data["action"][local_idx]
        episode_index = segment_data["episode_index"][local_idx]
        task_index = segment_data["task_index"][local_idx]
    else:
        # Fallback: direct indexing if it's a list/array
        state = segment_data[local_idx]["observation.state"]
        action = segment_data[local_idx]["action"]
        episode_index = segment_data[local_idx]["episode_index"]
        task_index = segment_data[local_idx]["task_index"]
    
    # Extract images
    # Dataset structure: observation.images.image and observation.images.wrist_image
    # Note: frames_data may contain multiple frames (if display_indices has multiple entries),
    # but we extract only the first frame (local_idx=0) which corresponds to the best match (start_idx).
    # This returns a SINGLE image per camera, not an array of images.
    images = {}
    if "frames_data" in segment:
        frames_data = segment["frames_data"]
        if isinstance(frames_data, dict):
            # Extract all camera images from the dataset
            # Keys are like "observation.images.image" and "observation.images.wrist_image"
            # frames_data[key] has shape (num_frames, H, W, C) if multiple frames, or (H, W, C) if single
            # We use local_idx=0 to get the first (best match) frame, resulting in shape (H, W, C)
            for key in frames_data:
                if key.startswith("observation.images."):
                    camera_name = key.replace("observation.images.", "")
                    img_array = frames_data[key]
                    # Handle both single frame and multi-frame cases
                    if img_array.ndim == 4:  # (num_frames, H, W, C)
                        images[camera_name] = img_array[local_idx]  # Extract single frame: (H, W, C)
                    elif img_array.ndim == 3:  # (H, W, C) - single frame already
                        images[camera_name] = img_array
                    else:
                        raise ValueError(f"Unexpected image array shape: {img_array.shape}")
        else:
            # Fallback: direct access (shouldn't happen with our dataset structure)
            if hasattr(frames_data, "__getitem__"):
                img = frames_data[local_idx] if hasattr(frames_data, "__len__") and len(frames_data) > local_idx else frames_data
                images["image"] = img if img.ndim == 3 else img[0]
    
    return {
        "state": state,
        "action": action,
        "images": images,
        "episode_index": int(episode_index),
        "task_index": int(task_index),
        "frame_idx": int(frame_idx),
    }


def map_camera_to_ricl_keys(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map LeRobot camera names to RICL expected keys.
    
    RICL expects: top_image, wrist_image, right_image
    This dataset has: image, wrist_image (no right_image)
    
    Mapping:
    - observation.images.image -> top_image
    - observation.images.wrist_image -> wrist_image
    - right_image -> duplicate of wrist_image (since dataset doesn't have it)
    """
    result = {}
    image_keys = list(images.keys())
    
    # Direct mapping for this dataset structure
    # Dataset has: "image" and "wrist_image" as camera names
    if "image" in images:
        result["top_image"] = images["image"]
    elif "wrist_image" in images:
        # Fallback: use wrist_image as top_image if no "image" key
        result["top_image"] = images["wrist_image"]
    elif image_keys:
        # Fallback: use first available image
        result["top_image"] = images[image_keys[0]]
    
    # Map wrist camera
    if "wrist_image" in images:
        result["wrist_image"] = images["wrist_image"]
    elif len(image_keys) > 1:
        # Use second image if available
        result["wrist_image"] = images[image_keys[1]]
    else:
        # Duplicate top_image if no wrist camera
        result["wrist_image"] = result.get("top_image")
    
    # Map right camera - dataset doesn't have this, so duplicate top_image
    # RICL expects right_image, so we provide it even if dataset doesn't have it
    if "right_image" in images:
        result["right_image"] = images["right_image"]
    elif len(image_keys) > 2:
        # Use third image if available
        result["right_image"] = images[image_keys[2]]
    elif "wrist_image" in images:
        result["right_image"] = images["wrist_image"]
    else:
        # Duplicate top_image (dataset doesn't have right_image)
        result["right_image"] = result.get("top_image")
    
    return result


def lerobot_retrieve_for_ricl(
    query_obs: dict[str, Any],
    handle: RetrieverHandle,
    knn_k: int,
    action_horizon: int,
    lamda: float,
    max_dist: float,
    *,
    use_text_retrieval: bool = False,
    text_query: str | None = None,
    top_k_frames: int = 128,
    overfetch_factor: int = 4,
) -> dict[str, Any]:
    """Retrieve demos using LeRobot retriever and format for RICL.
    
    This is Step 3: RICL-style wrapper around retrieve_rar.
    
    Args:
        query_obs: Query observation dict. Must contain:
            - query_top_image: numpy array or PIL Image for embedding
            - Optionally: query_state, query_wrist_image, etc. (passed through)
        handle: RetrieverHandle initialized with dataset and indices
        knn_k: Number of retrieved demos (num_retrieved_observations)
        action_horizon: Action horizon for action chunks
        lamda: Lambda parameter for distance weighting (exp(-λ * d))
        max_dist: Maximum distance for normalization
        use_text_retrieval: If True, use two-stage text+image retrieval
        text_query: Text query for task selection (if use_text_retrieval=True)
        top_k_frames: Number of frames to search (for image_retrieval)
        overfetch_factor: Overfetch factor for image_retrieval
    
    Returns:
        dict with keys:
            - retrieved_{i}_state: state array (single timestep, shape: [state_dim])
            - retrieved_{i}_top_image: single image array (shape: [H, W, C], NOT an array of images)
            - retrieved_{i}_wrist_image: single image array (shape: [H, W, C])
            - retrieved_{i}_right_image: single image array (shape: [H, W, C])
            - retrieved_{i}_actions: action chunk (shape: [action_horizon, action_dim])
            - retrieved_{i}_prompt: prompt string
            - exp_lamda_distances: (knn_k + 1, 1) array
            - inference_time: True
            - All original query_obs keys (passed through)
        
    Note: Each retrieved_{i}_*_image is a SINGLE image (not an array of images).
    The transform (RiclDroidInputs) will convert these to the dict format expected by RiclObservation.
    """
    more_obs = {"inference_time": True}
    
    # Extract query image
    query_image = query_obs.get("query_top_image")
    if query_image is None:
        # Try alternative keys
        for key in ["query_image", "top_image", "image"]:
            if key in query_obs:
                query_image = query_obs[key]
                break
    
    if query_image is None:
        raise ValueError("query_obs must contain 'query_top_image' or similar image key")
    
    # Get query embedding
    query_embedding = get_dino_image_embedding(query_image)
    
    # Perform retrieval
    if use_text_retrieval and text_query:
        # Two-stage: text -> task ranges, then image retrieval
        segments = retrieve_rar(
            handle,
            text_query,
            query_image,
            top_k_frames=top_k_frames,
            top_k_demos=knn_k,
            overfetch_factor=overfetch_factor,
            fetch_frames=True,
        )
    else:
        # Single-stage: image-only retrieval
        # We need to get all task ranges first, or search globally
        # For simplicity, get all task ranges
        all_task_indices = np.unique(handle.task_index_arr)
        task_frame_ranges = []
        for task_idx in all_task_indices:
            ranges = get_task_episode_indices(handle.task_index_arr, task_idx)
            task_frame_ranges.extend(ranges)
        
        if not task_frame_ranges:
            # Fallback: use entire dataset as one range
            task_frame_ranges = [(0, len(handle.hf_ds) - 1)]
        
        segments = image_retrieval(
            handle,
            query_embedding,
            task_frame_ranges,
            top_k_frames=top_k_frames,
            top_k_demos=knn_k,
            overfetch_factor=overfetch_factor,
            fetch_frames=True,
        )
    
    if len(segments) == 0:
        raise ValueError("No segments retrieved. Check your indices and dataset.")
    
    
    # Extract embeddings for distance computation
    retrieved_embeddings = []
    query_emb_np = query_embedding.detach().cpu().float().numpy()
    if query_emb_np.ndim == 1:
        query_emb_np = query_emb_np[None, :]
    
    # Process each retrieved segment
    for i, segment in enumerate(segments):
        # Extract single timestep (use start_idx, which is the best match)
        timestep_data = extract_single_timestep_from_segment(handle, segment, timestep_offset=0)
        
        # Map state
        more_obs[f"retrieved_{i}_state"] = timestep_data["state"]
        
        # Map images
        ricl_images = map_camera_to_ricl_keys(timestep_data["images"])
        # Ensure we have at least top_image
        if not ricl_images:
            # Fallback: create dummy image from state (shouldn't happen, but handle gracefully)
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            ricl_images = {"top_image": dummy_img, "wrist_image": dummy_img, "right_image": dummy_img}
        
        more_obs[f"retrieved_{i}_top_image"] = ricl_images.get("top_image")
        more_obs[f"retrieved_{i}_wrist_image"] = ricl_images.get("wrist_image", ricl_images.get("top_image"))
        more_obs[f"retrieved_{i}_right_image"] = ricl_images.get("right_image", ricl_images.get("top_image"))
        
        # Map actions
        # Get full action sequence from segment
        # segment["data"] is already sliced from rest_ds, which contains "action" column
        segment_data = segment["data"]
        if isinstance(segment_data, dict) and "action" in segment_data:
            # Dict format from HF dataset with_format("numpy")
            segment_actions = segment_data["action"]
            # segment_actions should be shape (num_frames, action_dim)
            if not isinstance(segment_actions, np.ndarray):
                segment_actions = np.array(segment_actions)
            # Ensure 2D: (num_frames, action_dim)
            if segment_actions.ndim == 1:
                segment_actions = segment_actions[None, :]
        else:
            # Fallback: construct from individual timesteps
            # This shouldn't happen with our current setup, but handle gracefully
            num_frames = segment["end_idx"] - segment["start_idx"] + 1
            # Try to get actions from the full dataset
            start_idx = segment["start_idx"]
            end_idx = segment["end_idx"] + 1
            rest_slice = handle.rest_ds[start_idx:end_idx]
            if isinstance(rest_slice, dict) and "action" in rest_slice:
                segment_actions = rest_slice["action"]
                if not isinstance(segment_actions, np.ndarray):
                    segment_actions = np.array(segment_actions)
            else:
                # Last resort: use single action repeated
                single_action = timestep_data["action"]
                if not isinstance(single_action, np.ndarray):
                    single_action = np.array(single_action)
                segment_actions = np.tile(single_action[None, :], (num_frames, 1))
        
        # Ensure 2D array: (num_frames, action_dim)
        if segment_actions.ndim == 1:
            segment_actions = segment_actions[None, :]
        
        local_step = timestep_data["frame_idx"] - segment["start_idx"]
        more_obs[f"retrieved_{i}_actions"] = get_action_chunk_at_inference_time(
            segment_actions,
            local_step,
            action_horizon,
        )
        
        # Map prompt
        task_idx = timestep_data["task_index"]
        if task_idx < len(handle.tasks_arr):
            more_obs[f"retrieved_{i}_prompt"] = str(handle.tasks_arr[task_idx])
        else:
            more_obs[f"retrieved_{i}_prompt"] = ""
        
        # Store embedding for distance computation
        # Recompute embedding from retrieved image
        retrieved_img = ricl_images.get("top_image")
        if retrieved_img is not None:
            retrieved_emb = get_dino_image_embedding(retrieved_img)
            retrieved_emb_np = retrieved_emb.detach().cpu().float().numpy()
            if retrieved_emb_np.ndim == 1:
                retrieved_emb_np = retrieved_emb_np[None, :]
            retrieved_embeddings.append(retrieved_emb_np[0])
        else:
            # Fallback: use zero embedding (shouldn't happen)
            retrieved_embeddings.append(np.zeros_like(query_emb_np[0]))
    
    # Compute distances and exp_lamda_distances
    if len(retrieved_embeddings) > 0:
        first_embedding = retrieved_embeddings[0]
        distances = [0.0]  # First retrieved demo has distance 0
        
        # Distances from first to other retrieved demos
        for emb in retrieved_embeddings[1:]:
            dist = np.linalg.norm(emb - first_embedding)
            distances.append(dist)
        
        # Distance from query to first retrieved demo
        query_dist = np.linalg.norm(query_emb_np[0] - first_embedding)
        distances.append(query_dist)
        
        # Normalize and compute exp(-λ * d)
        distances = np.clip(np.array(distances), 0, max_dist) / max_dist
        more_obs["exp_lamda_distances"] = np.exp(-lamda * distances).reshape(-1, 1)
    
    return {**query_obs, **more_obs}
