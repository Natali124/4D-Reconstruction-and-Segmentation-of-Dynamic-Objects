import torch
import numpy as np
import os
import pycolmap
from pathlib import Path
import imageio.v3 as iio
from PIL import Image
import sqlite3
import cv2
from moge.model.v2 import MoGeModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import urllib.request

def get_tracks_and_visibility(video, grid_size = 50, num_initializations = 5, device = 'cuda', verbose=True):
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)

    T = video.shape[1]
    init_frames = np.linspace(0, T-1, num_initializations, dtype=int).tolist()

    all_tracks = []
    all_visibility = []

    for i, init_frame_idx in enumerate(init_frames):
        if verbose:
            print(f"\nProcessing initialization frame {init_frame_idx} ({i+1}/{len(init_frames)})...")
        
        tracks, visibility = cotracker(
            video, 
            grid_size=grid_size,
            backward_tracking=True,
            grid_query_frame=init_frame_idx
        )
        
        all_tracks.append(tracks)
        all_visibility.append(visibility)
        if verbose:
            print(f"  Generated {tracks.shape[2]} tracks with bidirectional tracking")

    if len(all_tracks) > 0:
        combined_tracks = torch.cat(all_tracks, dim=2)  # Concatenate along track dimension
        combined_visibility = torch.cat(all_visibility, dim=2)
        
        if verbose:
            print(f"\nTotal combined tracks: {combined_tracks.shape[2]}")
            print(f"Video length: {T} frames")

    return combined_tracks, combined_visibility

def visualize_cotracker(video, pred_tracks, pred_visibility, filename, Visualizer):
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(video, pred_tracks, pred_visibility, filename=filename)

def get_3d_coordinates_for_frame(frames, model, device, frame_idx):
    """Get 3D coordinates and intrinsics for a specific frame using MoGe"""
    input_image = frames[frame_idx]
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    output = model.infer(input_image)
    return output["points"], output["mask"], output["intrinsics"]

def download_sam_checkpoint(model_type="vit_b", force_download=False):
    """
    Download SAM checkpoint if it doesn't exist
    
    Args:
        model_type: "vit_b", "vit_l", or "vit_h"
        force_download: whether to download even if file exists
        
    Returns:
        Path to the checkpoint file
    """
    checkpoint_urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    checkpoint_names = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth"
    }
    
    if model_type not in checkpoint_urls:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(checkpoint_urls.keys())}")
    
    checkpoint_path = checkpoint_names[model_type]
    
    if not os.path.exists(checkpoint_path) or force_download:
        print(f"Downloading SAM {model_type} checkpoint...")
        print(f"This may take a few minutes depending on your internet connection.")
        
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\rDownloading: {percent}% ({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)", end="")
        
        try:
            urllib.request.urlretrieve(checkpoint_urls[model_type], checkpoint_path, progress_hook)
            print(f"\nDownload completed: {checkpoint_path}")
        except Exception as e:
            print(f"\nError downloading checkpoint: {e}")
            raise
    else:
        print(f"Using existing checkpoint: {checkpoint_path}")
    
    return checkpoint_path

def segment_frame_with_sam(frame, model_type="vit_b", checkpoint_path=None, device="cuda", verbose=True):
    """
    Segment frame using SAM model from Meta AI
    
    Args:
        frame: numpy array of shape (H, W, 3)
        model_type: SAM model type ("vit_b", "vit_l", "vit_h")
        checkpoint_path: path to SAM checkpoint (auto-download if None)
        device: device to run on
        verbose: print progress
        
    Returns:
        Dictionary mapping object_id -> mask (boolean array of shape (H, W))
    """
    if verbose:
        print(f"Running SAM segmentation on frame with model {model_type}...")
    
    # Auto-download checkpoint if needed
    if checkpoint_path is None:
        checkpoint_path = download_sam_checkpoint(model_type)
    
    # Load SAM model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    # Convert to RGB if needed (SAM expects RGB)
    if frame.shape[-1] == 3:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.dtype == np.uint8 else frame
    else:
        frame_rgb = frame
    
    # Generate masks
    masks = mask_generator.generate(frame_rgb)
    
    # Sort masks by area (largest first)
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    object_masks = {}
    
    for i, mask_data in enumerate(masks):
        object_masks[i] = mask_data['segmentation'].astype(bool)
    
    if verbose:
        print(f"Found {len(object_masks)} segments")
        if len(masks) > 0:
            print(f"Largest segment area: {masks[0]['area']}")
            print(f"Smallest segment area: {masks[-1]['area']}")
    
    return object_masks

def check_tracks_on_same_object(track_ids, tracks_frame0, object_masks, verbose=False):
    """
    Check if all tracks are on the same object in frame 0
    
    Args:
        track_ids: list of track IDs to check
        tracks_frame0: numpy array of track positions in frame 0, shape (N, 2)
        object_masks: dict mapping object_id -> mask from SAM
        verbose: print debug info
        
    Returns:
        bool: True if all tracks are on the same object, False otherwise
    """
    if len(object_masks) == 0:
        return False
    
    # Find which object each track belongs to
    track_objects = []
    
    for track_id in track_ids:
        track_pos = tracks_frame0[track_id]
        x, y = int(np.clip(track_pos[0], 0, list(object_masks.values())[0].shape[1] - 1)), \
               int(np.clip(track_pos[1], 0, list(object_masks.values())[0].shape[0] - 1))
        
        # Check which object this track belongs to
        track_object = None
        for object_id, mask in object_masks.items():
            if mask[y, x]:  # Note: mask is indexed as [y, x]
                track_object = object_id
                break
        
        if track_object is None:
            if verbose:
                print(f"Track {track_id} at ({x}, {y}) not on any object")
            return False
        
        track_objects.append(track_object)
    
    # Check if all tracks are on the same object
    if len(set(track_objects)) == 1:
        if verbose:
            print(f"All {len(track_ids)} tracks are on object {track_objects[0]}")
        return True
    else:
        if verbose:
            print(f"Tracks span {len(set(track_objects))} different objects: {set(track_objects)}")
        return False

def check_tracks_visible_in_all_frames(track_ids, visibility_np, frame_pairs):
    """
    Check if all tracks are visible in all frames involved in frame pairs
    
    Args:
        track_ids: list of track IDs to check
        visibility_np: numpy array of visibility, shape (T, N)
        frame_pairs: list of (frame1, frame2) tuples
        
    Returns:
        bool: True if all tracks are visible in all relevant frames
    """
    # Get all unique frame indices
    all_frames = set()
    for frame1, frame2 in frame_pairs:
        all_frames.add(frame1)
        all_frames.add(frame2)
    
    # Check visibility for each track in each frame
    for track_id in track_ids:
        for frame_idx in all_frames:
            if visibility_np[frame_idx, track_id] <= 0.5:
                return False
    
    return True

def convert_moge_intrinsics_to_opencv(intrinsics, image_height, image_width):
    """
    Convert MoGe intrinsics (normalized coordinates) to OpenCV camera matrix
    
    MoGe intrinsics are in normalized coordinates where:
    - cx, cy are in [0, 1] range
    - fx, fy are focal lengths in normalized coordinates
    
    OpenCV expects pixel coordinates
    """
    # Extract intrinsic parameters
    fx_norm = intrinsics[0, 0].item()
    fy_norm = intrinsics[1, 1].item()
    cx_norm = intrinsics[0, 2].item()
    cy_norm = intrinsics[1, 2].item()
    
    # Convert to pixel coordinates
    fx_pixel = fx_norm * image_width
    fy_pixel = fy_norm * image_height
    cx_pixel = cx_norm * image_width
    cy_pixel = cy_norm * image_height
    
    # Create OpenCV camera matrix
    camera_matrix = np.array([
        [fx_pixel, 0, cx_pixel],
        [0, fy_pixel, cy_pixel],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return camera_matrix

def extract_track_correspondences_fast(frames, model, device, pred_tracks, pred_visibility, 
                                     frame_pairs, min_depth=0.1, max_depth=100.0, verbose=True):
    """
    Optimized version of extract_track_correspondences using vectorized operations
    Modified to work with frame 0 to all other frames
    """
    # Batch process 3D coordinates for all reference frames (now just frame 0)
    frame_3d_data = {}
    
    # Get unique reference frames to avoid duplicate processing
    unique_ref_frames = list(set([pair[0] for pair in frame_pairs]))
    
    if verbose:
        print(f"Processing 3D coordinates for {len(unique_ref_frames)} reference frames...")
    
    for frame_idx in unique_ref_frames:
        points_3d, mask, intrinsics = get_3d_coordinates_for_frame(frames, model, device, frame_idx)
        frame_3d_data[frame_idx] = {
            'points_3d': points_3d.cpu().numpy(),  # Move to CPU once
            'mask': mask.cpu().numpy(),
            'intrinsics': intrinsics
        }
    
    # Convert tracks to numpy for faster indexing
    tracks_np = pred_tracks[0].cpu().numpy()  # Shape: (T, N)
    visibility_np = pred_visibility[0].cpu().numpy()  # Shape: (T, N)
    
    num_tracks = tracks_np.shape[1]
    track_correspondences = {}
    
    if verbose:
        print(f"Extracting correspondences for {num_tracks} tracks across {len(frame_pairs)} frame pairs...")
    
    # Process all tracks at once for each frame pair
    for frame1_idx, frame2_idx in frame_pairs:
        # Get visibility masks for both frames (all tracks at once)
        vis_frame1 = visibility_np[frame1_idx] > 0.5  # Shape: (N,)
        vis_frame2 = visibility_np[frame2_idx] > 0.5  # Shape: (N,)
        visible_both = vis_frame1 & vis_frame2
        
        if not np.any(visible_both):
            continue
            
        # Get track positions for visible tracks
        tracks_frame1 = tracks_np[frame1_idx, visible_both]  # Shape: (M, 2)
        tracks_frame2 = tracks_np[frame2_idx, visible_both]  # Shape: (M, 2)
        visible_track_ids = np.where(visible_both)[0]
        
        # Get 3D data for reference frame
        points_3d_frame = frame_3d_data[frame1_idx]['points_3d']
        mask_frame = frame_3d_data[frame1_idx]['mask']
        
        # Vectorized pixel coordinate extraction
        cols = np.clip(tracks_frame1[:, 0].astype(int), 0, points_3d_frame.shape[1] - 1)
        rows = np.clip(tracks_frame1[:, 1].astype(int), 0, points_3d_frame.shape[0] - 1)
        
        # Check mask validity (vectorized)
        valid_mask = mask_frame[rows, cols]
        
        # Extract 3D points for valid tracks
        points_3d = points_3d_frame[rows[valid_mask], cols[valid_mask]]
        
        # Filter by depth and finite values (vectorized)
        depth_mask = (points_3d[:, 2] >= min_depth) & (points_3d[:, 2] <= max_depth)
        finite_mask = np.isfinite(points_3d).all(axis=1)
        non_zero_mask = ~np.all(np.abs(points_3d) < 1e-6, axis=1)  # Check if points are not essentially zero
        final_mask = depth_mask & finite_mask & non_zero_mask
        
        if not np.any(final_mask):
            continue
        
        # Get final valid data
        valid_indices = np.where(valid_mask)[0]
        final_valid_indices = valid_indices[final_mask]
        final_track_ids = visible_track_ids[final_valid_indices]
        final_points_3d = points_3d[final_mask]
        final_points_2d = tracks_frame2[final_valid_indices]
        
        # Check 2D points are finite
        finite_2d_mask = np.isfinite(final_points_2d).all(axis=1)
        final_track_ids = final_track_ids[finite_2d_mask]
        final_points_3d = final_points_3d[finite_2d_mask]
        final_points_2d = final_points_2d[finite_2d_mask]
        
        # Store correspondences
        for i, track_id in enumerate(final_track_ids):
            if track_id not in track_correspondences:
                track_correspondences[track_id] = []
            track_correspondences[track_id].append(
                (int(frame1_idx), int(frame2_idx), final_points_3d[i], final_points_2d[i])
            )
    
    if verbose:
        total_correspondences = sum(len(v) for v in track_correspondences.values())
        print(f"Found {len(track_correspondences)} tracks with {total_correspondences} total correspondences")
    
    return track_correspondences

def solve_frame_to_zero_pnp_ransac(track_correspondences, camera_matrices, frame_pairs,
                                   max_iterations=100, reprojection_threshold=8.0, 
                                   min_tracks=6, verbose=True, use_sam=False,
                                   frames=None, pred_tracks=None, pred_visibility=None):
    """
    Global RANSAC: Sample tracks once, solve PnP for all frame pairs, average errors across frame pairs
    """
    if verbose:
        print(f'🚀 Beginning RANSAC with {len(track_correspondences)} tracks, {len(frame_pairs)} frame pairs')
    
    # Get all unique track IDs that appear in correspondences
    all_track_ids = list(track_correspondences.keys())
    
    # Pre-filter tracks: only keep tracks that are visible in ALL relevant frames
    if pred_visibility is not None:
        visibility_np = pred_visibility[0].cpu().numpy()  # Shape: (T, N)
        
        # Get all unique frame indices from frame pairs
        all_relevant_frames = set()
        for frame1, frame2 in frame_pairs:
            all_relevant_frames.add(frame1)
            all_relevant_frames.add(frame2)
        
        # Filter tracks to only those visible in all relevant frames
        globally_visible_tracks = []
        for track_id in all_track_ids:
            visible_in_all = True
            for frame_idx in all_relevant_frames:
                if frame_idx < visibility_np.shape[0] and visibility_np[frame_idx, track_id] <= 0.5:
                    visible_in_all = False
                    break
            if visible_in_all:
                globally_visible_tracks.append(track_id)
        
        if verbose:
            print(f"📊 Pre-filtering tracks: {len(all_track_ids)} total -> {len(globally_visible_tracks)} globally visible")
        
        # Keep all tracks that have correspondences (they should already be properly filtered)
        all_track_ids = list(track_correspondences.keys())
    
    # SAM segmentation for frame 0
    object_masks = {}
    
    if use_sam:
        if frames is None or pred_tracks is None or pred_visibility is None:
            if verbose:
                print("⚠️ Warning: sam requires frames, pred_tracks, and pred_visibility for special sampling")
            use_sam = False  # Fall back to normal sampling
        else:
            object_masks = segment_frame_with_sam(frames[0], model_type="vit_b", verbose=verbose)
            tracks_frame0 = pred_tracks[0].cpu().numpy()[0]  # Shape: (N, 2)  
    
    # Organize correspondences by frame pair for faster lookup
    frame_pair_correspondences = {}
    for frame1_idx, frame2_idx in frame_pairs:
        frame_pair_correspondences[(frame1_idx, frame2_idx)] = {
            'track_ids': [],
            'points_3d': [],
            'points_2d': []
        }
    
    # Populate frame pair correspondences
    for track_id, correspondences in track_correspondences.items():
        for corr_frame1, corr_frame2, point_3d, point_2d in correspondences:
            pair_key = (corr_frame1, corr_frame2)
            if pair_key in frame_pair_correspondences:
                frame_pair_correspondences[pair_key]['track_ids'].append(track_id)
                frame_pair_correspondences[pair_key]['points_3d'].append(point_3d)
                frame_pair_correspondences[pair_key]['points_2d'].append(point_2d)
    
    # Convert to numpy arrays and validate - WITH DETAILED DEBUGGING
    valid_frame_pairs = []
    print(f"\n🔍 DETAILED FRAME PAIR ANALYSIS:")
    print(f"{'Frame Pair':<15} {'Tracks':<8} {'Status':<20} {'Issue':<30}")
    print("-" * 80)
    
    for pair_idx, pair_key in enumerate(frame_pair_correspondences):
        num_tracks = len(frame_pair_correspondences[pair_key]['track_ids'])
        frame1, frame2 = pair_key
        
        if num_tracks >= min_tracks:
            frame_pair_correspondences[pair_key]['points_3d'] = np.array(
                frame_pair_correspondences[pair_key]['points_3d'], dtype=np.float32
            )
            frame_pair_correspondences[pair_key]['points_2d'] = np.array(
                frame_pair_correspondences[pair_key]['points_2d'], dtype=np.float32
            )
            valid_frame_pairs.append(pair_key)
            status = "✅ VALID"
            issue = ""
        else:
            status = "❌ INVALID"
            issue = f"Only {num_tracks}/{min_tracks} tracks"
        
        print(f"({frame1:2d},{frame2:2d})        {num_tracks:<8} {status:<20} {issue:<30}")
        
        # Extra debugging for frames that suddenly lose tracks
        if pair_idx > 0 and num_tracks == 0:
            prev_pair = list(frame_pair_correspondences.keys())[pair_idx-1]
            prev_tracks = len(frame_pair_correspondences[prev_pair]['track_ids'])
            if prev_tracks > 0:
                print(f"  🚨 SUDDEN TRACK LOSS: Previous pair had {prev_tracks} tracks, this has {num_tracks}")
        
        # Show a pattern of declining tracks
        if pair_idx >= 5:  # After first few pairs
            recent_pairs = list(frame_pair_correspondences.keys())[max(0, pair_idx-3):pair_idx+1]
            recent_counts = [len(frame_pair_correspondences[p]['track_ids']) for p in recent_pairs]
            if all(c < min_tracks for c in recent_counts[-3:]) and recent_counts[0] >= min_tracks:
                print(f"  📉 TRACK DECLINE PATTERN: {recent_counts} (last 4 pairs)")
    
    print("-" * 80)
    print(f"📈 SUMMARY: {len(valid_frame_pairs)}/{len(frame_pairs)} frame pairs are valid for PnP")
    
    # Show detailed breakdown of where tracks disappear
    if len(valid_frame_pairs) < len(frame_pairs):
        print(f"\n🔍 TRACK LOSS ANALYSIS:")
        
        # Find where tracks start disappearing
        track_counts_by_frame = {}
        for (f1, f2), data in frame_pair_correspondences.items():
            track_counts_by_frame[f2] = len(data['track_ids'])
        
        sorted_frames = sorted(track_counts_by_frame.keys())
        print(f"Track count by target frame:")
        for frame_idx in sorted_frames:
            count = track_counts_by_frame[frame_idx]
            status = "✅" if count >= min_tracks else "❌"
            print(f"  Frame {frame_idx:2d}: {count:3d} tracks {status}")
            
            # Identify the cutoff point
            if frame_idx > sorted_frames[0] and count < min_tracks:
                prev_frame = sorted_frames[sorted_frames.index(frame_idx) - 1]
                prev_count = track_counts_by_frame[prev_frame]
                if prev_count >= min_tracks:
                    print(f"  🚨 CRITICAL CUTOFF: Frame {prev_frame} had {prev_count} tracks, Frame {frame_idx} has {count}")
                    print(f"     This explains why poses become None starting around frame {frame_idx}")
    
    if len(all_track_ids) < min_tracks:
        print(f"❌ FATAL ERROR: Only {len(all_track_ids)} tracks available (need {min_tracks})")
        return [], None, {}
    
    if len(valid_frame_pairs) == 0:
        print(f"❌ FATAL ERROR: No frame pairs have enough tracks for PnP")
        return [], None, {}
    
    best_poses = None
    best_inlier_count = 0
    best_global_track_errors = None
    best_global_inliers = None
    
    # Initialize poses list to match frame_pairs length
    initial_poses = [(None, None)] * len(frame_pairs)
    
    print(f"\n🎯 Starting RANSAC with {max_iterations} iterations...")
    
    for iteration in range(max_iterations):
        # Step 1: Sample subset of tracks from available tracks
        sample_size = min_tracks
        
        if use_sam and len(object_masks) > 0:
            # Special sampling with object constraint
            max_sampling_attempts = 50
            sampling_success = False
            
            for attempt in range(max_sampling_attempts):
                sampled_track_ids = np.random.choice(all_track_ids, size=sample_size, replace=False)
                
                if not check_tracks_on_same_object(sampled_track_ids, tracks_frame0, object_masks, verbose=False):
                    continue
                
                sampling_success = True
                break
            
            if not sampling_success:
                if verbose and iteration == 0:
                    print(f"⚠️ Warning: Could not find valid track sample after {max_sampling_attempts} attempts")
                continue
            
            if verbose and iteration % 50 == 0:
                print(f"  Iteration {iteration}: Found valid sample after {attempt + 1} attempts")
        else:
            # Normal sampling
            sampled_track_ids = np.random.choice(all_track_ids, size=sample_size, replace=False)
        
        sampled_track_set = set(sampled_track_ids)
        
        # Step 2: For each frame pair, solve PnP using only the sampled tracks
        current_poses = initial_poses.copy()  # Start with None values
        successful_pairs = 0
        failed_pairs = []
        
        for pair_idx, (frame1_idx, frame2_idx) in enumerate(frame_pairs):
            pair_key = (frame1_idx, frame2_idx)
            
            if pair_key not in frame_pair_correspondences:
                failed_pairs.append((pair_idx, frame2_idx, "No correspondences"))
                continue
                
            pair_data = frame_pair_correspondences[pair_key]
            
            # Find which of our sampled tracks are available in this frame pair
            available_indices = [i for i, track_id in enumerate(pair_data['track_ids']) 
                               if track_id in sampled_track_set]
            
            if len(available_indices) < 6:  # Need at least 6 points for PnP
                failed_pairs.append((pair_idx, frame2_idx, f"Only {len(available_indices)}/6 sampled tracks available"))
                continue
            
            # Get points for sampled tracks in this frame pair
            pair_points_3d = pair_data['points_3d'][available_indices]
            pair_points_2d = pair_data['points_2d'][available_indices]
            camera_matrix = camera_matrices[frame2_idx]
            
            # Solve PnP for this frame pair
            try:
                success, rvec, tvec = cv2.solvePnP(
                    pair_points_3d,
                    pair_points_2d,
                    camera_matrix,
                    np.zeros(4, dtype=np.float32),
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.flatten()
                    current_poses[pair_idx] = (R, t)
                    successful_pairs += 1
                else:
                    failed_pairs.append((pair_idx, frame2_idx, "PnP solver failed"))
            except cv2.error as e:
                failed_pairs.append((pair_idx, frame2_idx, f"CV2 error: {str(e)[:50]}"))
                continue
        
        # Show detailed failure analysis for first iteration
        if iteration == 0 and failed_pairs:
            print(f"\n🚨 FIRST ITERATION FAILURE ANALYSIS:")
            print(f"{'Pair Idx':<10} {'Frame':<8} {'Reason':<50}")
            print("-" * 70)
            for pair_idx, frame_idx, reason in failed_pairs:
                print(f"{pair_idx:<10} {frame_idx:<8} {reason:<50}")
            print(f"Total successful pairs: {successful_pairs}/{len(frame_pairs)}")
        
        # Step 3: Apply each R_i, t_i to ALL tracks and average errors across frame pairs
        if successful_pairs == 0:
            if iteration == 0:
                print(f"❌ CRITICAL: No successful pairs in first iteration - all poses will be None!")
            continue
        
        track_errors_across_pairs = {}  # track_id -> list of errors from different frame pairs
        
        for pair_idx, (frame1_idx, frame2_idx) in enumerate(frame_pairs):
            R, t = current_poses[pair_idx]
            if R is None:
                continue
                
            pair_key = (frame1_idx, frame2_idx)
            pair_data = frame_pair_correspondences[pair_key]
            camera_matrix = camera_matrices[frame2_idx]
            
            # Project ALL tracks in this frame pair (not just sampled ones)
            if len(pair_data['points_3d']) > 0:
                try:
                    rvec, _ = cv2.Rodrigues(R)
                    tvec = t.reshape(3, 1)
                    
                    projected_points, _ = cv2.projectPoints(
                        pair_data['points_3d'],
                        rvec,
                        tvec,
                        camera_matrix,
                        np.zeros(4, dtype=np.float32)
                    )
                    projected_points = projected_points.reshape(-1, 2)
                    
                    # Compute errors for all tracks in this frame pair
                    errors = np.linalg.norm(projected_points - pair_data['points_2d'], axis=1)
                    
                    # Store errors for each track
                    for i, track_id in enumerate(pair_data['track_ids']):
                        if track_id not in track_errors_across_pairs:
                            track_errors_across_pairs[track_id] = []
                        track_errors_across_pairs[track_id].append(errors[i])
                except Exception as e:
                    if verbose and iteration == 0:
                        print(f"⚠️ Projection error for pair {pair_key}: {e}")
                    continue
        
        # Step 4: Average errors across frame pairs for each track and classify
        current_global_track_errors = {}
        current_global_inliers = set()
        
        for track_id, error_list in track_errors_across_pairs.items():
            if error_list:
                avg_error = np.mean(error_list)
                current_global_track_errors[track_id] = avg_error
                if avg_error <= reprojection_threshold:
                    current_global_inliers.add(track_id)
        
        current_inlier_count = len(current_global_inliers)
        
        # Update best solution
        if current_inlier_count > best_inlier_count:
            best_inlier_count = current_inlier_count
            best_poses = current_poses.copy()
            best_global_track_errors = current_global_track_errors.copy()
            best_global_inliers = current_global_inliers.copy()
            
            if verbose and iteration % 50 == 0:
                print(f"  Iteration {iteration}: {current_inlier_count} inlier tracks, {successful_pairs} successful pairs")
    
    if best_poses is None:
        print(f"❌ FATAL ERROR: No valid poses found in any RANSAC iteration")
        return [], None, {}

    # FINAL REFINEMENT: Re-estimate poses using ALL inlier tracks instead of just the 6 sampled ones
    print(f"\n🔧 FINAL POSE REFINEMENT: Re-estimating poses using all {len(best_global_inliers)} inlier tracks...")
    
    refined_poses = []
    refinement_successful = 0
    
    for pair_idx, (frame1_idx, frame2_idx) in enumerate(frame_pairs):
        pair_key = (frame1_idx, frame2_idx)
        
        if pair_key not in frame_pair_correspondences:
            refined_poses.append((None, None))
            continue
            
        pair_data = frame_pair_correspondences[pair_key]
        
        # Find which inlier tracks are available in this frame pair
        inlier_indices = [i for i, track_id in enumerate(pair_data['track_ids']) 
                         if track_id in best_global_inliers]
        
        if len(inlier_indices) < 6:  # Still need minimum 6 points for PnP
            refined_poses.append((None, None))
            continue
        
        # Get points for ALL inlier tracks in this frame pair
        inlier_points_3d = pair_data['points_3d'][inlier_indices]
        inlier_points_2d = pair_data['points_2d'][inlier_indices]
        camera_matrix = camera_matrices[frame2_idx]
        
        # Re-solve PnP using all inlier tracks
        try:
            success, rvec, tvec = cv2.solvePnP(
                inlier_points_3d,
                inlier_points_2d,
                camera_matrix,
                np.zeros(4, dtype=np.float32),
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()
                refined_poses.append((R, t))
                refinement_successful += 1
            else:
                refined_poses.append((None, None))
        except cv2.error as e:
            if verbose:
                print(f"⚠️ Refinement failed for pair {pair_key}: {e}")
            refined_poses.append((None, None))
    
    print(f"🎯 REFINEMENT RESULTS: {refinement_successful}/{len(frame_pairs)} poses successfully refined")
    
    # Compare before/after
    original_valid = sum(1 for R, t in best_poses if R is not None)
    refined_valid = sum(1 for R, t in refined_poses if R is not None)
    print(f"📊 IMPROVEMENT: {original_valid} -> {refined_valid} valid poses ({refined_valid - original_valid:+d})")
    
    # Use refined poses as final result
    frame_pair_results = refined_poses
    global_outlier_tracks = set(all_track_ids) - best_global_inliers
    
    if verbose:
        valid_poses = sum(1 for R, t in best_poses if R is not None)
        print(f"\n🎯 FINAL GLOBAL RANSAC RESULTS:")
        print(f"  ✅ Valid poses found: {valid_poses}/{len(frame_pairs)} frame pairs")
        print(f"  🎯 Global inlier tracks: {len(best_global_inliers)}")
        print(f"  ❌ Global outlier tracks: {len(global_outlier_tracks)}")
        print(f"  🏆 Best inlier count: {best_inlier_count}")
        
        # Show which frames have valid poses with detailed breakdown
        valid_frame_indices = []
        none_frame_indices = []
        for i, (R, t) in enumerate(best_poses):
            target_frame = frame_pairs[i][1]
            if R is not None:
                valid_frame_indices.append(target_frame)
            else:
                none_frame_indices.append(target_frame)
        
        print(f"  ✅ Frames with valid poses: {valid_frame_indices}")
        print(f"  ❌ Frames with None poses: {none_frame_indices}")
        
        if none_frame_indices:
            print(f"\n🔍 ROOT CAUSE ANALYSIS:")
            print(f"  The None poses start appearing at frame {min(none_frame_indices)}")
            print(f"  This corresponds to the point where tracks become insufficient for PnP")
            print(f"  Check the track visibility/correspondence extraction for these later frames")
    
    return frame_pair_results, None, {
        'frame_pairs': frame_pairs,
        'individual_errors': [0] * len(frame_pairs),  # Not meaningful in global RANSAC
        'weights': [1 if poses[0] is not None else 0 for poses in frame_pair_results],
        'valid_pairs': sum(1 for R, t in frame_pair_results if R is not None),
        'global_inlier_tracks': best_global_inliers,
        'global_outlier_tracks': global_outlier_tracks,
        'global_track_errors': best_global_track_errors
    }



def main_pose_estimation_frame_zero_to_all(frames, model, device, pred_tracks, pred_visibility, 
                                         result_frequency=1, ransac_max_iters=100, 
                                         ransac_threshold=8.0, verbose=True, use_sam=False):
    """
    Main function for camera pose estimation from frame 0 to all other frames
    """
    # Create frame pairs from frame 0 to all other frames
    frame_pairs = []
    for frame2_idx in range(result_frequency, len(frames), result_frequency):
        if frame2_idx >= len(frames):
            break
        frame_pairs.append((0, frame2_idx))  # Always from frame 0
    
    if verbose:
        print(f"Processing {len(frame_pairs)} frame pairs from frame 0:")
        print(f"Frame pairs: {frame_pairs[:10]}{'...' if len(frame_pairs) > 10 else ''}")
    
    # Extract track correspondences using existing optimized function
    track_correspondences = extract_track_correspondences_fast(
        frames, model, device, pred_tracks, pred_visibility, frame_pairs, verbose=verbose
    )
    
    if len(track_correspondences) == 0:
        print("Error: No valid track correspondences found")
        return None, None, None
    
    if verbose:
        print(f"Found correspondences for {len(track_correspondences)} tracks")
        # Debug: check correspondence distribution
        correspondence_counts = {}
        for track_id, corrs in track_correspondences.items():
            correspondence_counts[len(corrs)] = correspondence_counts.get(len(corrs), 0) + 1
        print(f"Correspondence distribution: {dict(sorted(correspondence_counts.items()))}")
    
    # Prepare camera matrices for all target frames
    camera_matrices = {}
    image_height, image_width = frames[0].shape[:2]
    
    unique_target_frames = list(set([pair[1] for pair in frame_pairs]))
    # Also include frame 0 for consistency
    unique_target_frames.append(0)
    unique_target_frames = list(set(unique_target_frames))
    
    if verbose:
        print(f"Processing camera matrices for {len(unique_target_frames)} target frames...")
    
    for frame_idx in unique_target_frames:
        _, _, intrinsics = get_3d_coordinates_for_frame(frames, model, device, frame_idx)
        camera_matrices[frame_idx] = convert_moge_intrinsics_to_opencv(
            intrinsics, image_height, image_width
        )
    
    # Solve frame-to-zero PnP RANSAC
    pose_results, weighted_error, details = solve_frame_to_zero_pnp_ransac(
        track_correspondences, camera_matrices, frame_pairs,
        max_iterations=ransac_max_iters, 
        reprojection_threshold=ransac_threshold,
        verbose=verbose, use_sam=use_sam,
        frames=frames, pred_tracks=pred_tracks, pred_visibility=pred_visibility
    )
    
    return pose_results, weighted_error, {
        'track_correspondences': track_correspondences,
        'frame_pair_details': details,
        'camera_matrices': camera_matrices,
        'frame_pairs': frame_pairs,  # Include this for proper indexing later
    }

def main(device='cuda', frames=None, url=None,
         grid_size=50, num_initializations=5, ransac_threshold=8.0, result_frequency=1, 
         verbose=True, visualize_cotracker=False, long_output=True, pred_tracks=None, 
         pred_visibility=None, use_sam=False):
    """
    Main function for camera pose estimation with debugging
    """
    # Load video from url
    if frames is None or len(frames) == 0:
        if url:
            frames = iio.imread(url, plugin="FFMPEG")
        else:
            raise ValueError("Either frames or url must be provided")
    
    # Limit frames for testing
    frames = frames[:50]
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

    if verbose:
        print(f"Video shape: {video.shape}, Frames: {len(frames)}")

    # Run cotracker
    if pred_tracks is None:
        pred_tracks, pred_visibility = get_tracks_and_visibility(
            video, grid_size=grid_size, num_initializations=num_initializations, 
            device=device, verbose=verbose
        )
    
    if verbose:
        print(f"Tracks shape: {pred_tracks.shape}")
        print(f"Visibility shape: {pred_visibility.shape}")

    # Load MoGe
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)   

    # Estimate camera pose using frame 0 to all others approach
    if verbose:
        print(f"Estimating pose from frame 0 to all other frames")
        print(f"Result frequency: {result_frequency}, RANSAC threshold: {ransac_threshold}")
    
    R, t, results = main_pose_estimation_frame_zero_to_all(
        frames, model, device, pred_tracks, pred_visibility, 
        result_frequency=result_frequency, 
        ransac_threshold=ransac_threshold, verbose=verbose, use_sam=use_sam
    )

    all_results = {('track_based', 'frame_zero_to_all'): (R, t, results)}

    if verbose:
        print(f"Final results summary:")
        if R is not None:
            valid_poses = sum(1 for pose in R if pose[0] is not None)
            print(f"  Valid poses: {valid_poses}/{len(R)}")
        else:
            print(f"  No poses found!")

    if long_output:
        return all_results, frames, pred_tracks, pred_visibility
    else:
        return all_results