import torch

def get_3d_coordinates_for_frame(frames, model, device, frame_idx):
    """Get 3D coordinates and intrinsics for a specific frame using MoGe"""
    input_image = frames[frame_idx]
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    output = model.infer(input_image)
    return output["points"], output["mask"], output["intrinsics"]

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

def create_hierarchical_outliers_gif(frames, pred_tracks, pred_visibility, outlier_tracks,
                                         output_path='hierarchical_outliers_fast.gif', 
                                         duration=300, skip_frames=1, max_frames=None):
    """
    Fast version of hierarchical outlier GIF with HUGE TITLE and BIG POINTS, NO LEGEND
    
    Args:
        frames: Array of video frames
        pred_tracks: Predicted tracks tensor
        pred_visibility: Predicted visibility tensor  
        outlier_tracks: List of sets, where outlier_tracks[i+1] ⊆ outlier_tracks[i]
        output_path: Output GIF file path
        duration: Duration per frame in milliseconds
        skip_frames: Skip every N frames (1 = use all)
        max_frames: Maximum frames to process
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import io
    
    # Validation
    if frames is None or pred_tracks is None or pred_visibility is None:
        print("Error: Invalid input data")
        return
    
    if not outlier_tracks or len(outlier_tracks) == 0:
        print("Error: No outlier_tracks provided")
        return
    
    # Convert tensors once
    if hasattr(pred_tracks, 'cpu'):
        tracks_np = pred_tracks[0].cpu().numpy()
        visibility_np = pred_visibility[0].cpu().numpy()
    else:
        tracks_np = pred_tracks[0] if pred_tracks.ndim == 4 else pred_tracks
        visibility_np = pred_visibility[0] if pred_visibility.ndim == 3 else pred_visibility
    
    # Pre-compute colors and assignments
    colors = plt.cm.Set1(np.linspace(0, 1, len(outlier_tracks)))
    track_colors = {}
    track_levels = {}
    
    # Assign colors (most specific level wins)
    for level in range(len(outlier_tracks) - 1, -1, -1):
        for track_id in outlier_tracks[level]:
            if track_id not in track_colors:
                track_colors[track_id] = colors[level]
                track_levels[track_id] = level
    
    print(f"Fast hierarchical GIF: {len(track_colors)} classified tracks, {len(outlier_tracks)} levels")
    
    # Frame selection
    frame_indices = list(range(0, len(frames), skip_frames))
    if max_frames:
        frame_indices = frame_indices[:max_frames]
    
    # FIXED SIZING - Use consistent approach
    frame_height, frame_width = frames[0].shape[:2]
    
    # Calculate figure size to maintain aspect ratio and good resolution
    target_width = max(10, frame_width / 80)  # Target figure width
    target_height = target_width * (frame_height / frame_width)
    
    # Use consistent high DPI and sizing with HUGE title and BIG points
    dpi = 100
    figsize = (target_width, target_height)
    fontsize_title = 48  # HUGE TITLE
    marker_size = 150    # VERY BIG POINTS
    
    # Turn off interactive plotting
    plt.ioff()
    
    gif_frames = []
    
    for i, frame_idx in enumerate(frame_indices):
        if i % 10 == 0:
            print(f"Processing frame {i+1}/{len(frame_indices)}")
        
        try:
            # Create figure with consistent sizing and orientation
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            # Ensure consistent image display with proper origin
            ax.imshow(frames[frame_idx], origin='upper', aspect='equal')
            ax.set_title(f'Frame {frame_idx}', fontsize=fontsize_title, pad=20)
            ax.axis('off')
            
            # Set consistent axis limits to prevent rotation/scaling issues
            ax.set_xlim(0, frame_width)
            ax.set_ylim(frame_height, 0)  # Invert Y-axis to match image coordinates
            
            plt.tight_layout(pad=0.5)
            
            # Get visible tracks for this frame
            if frame_idx < visibility_np.shape[0]:
                visible_mask = visibility_np[frame_idx, :] > 0.5
                visible_track_ids = np.where(visible_mask)[0]
                
                # Filter to classified tracks that are visible
                classified_visible = [tid for tid in visible_track_ids if tid in track_colors]
                
                if classified_visible:
                    # Get positions for classified visible tracks
                    positions = tracks_np[frame_idx, classified_visible]
                    
                    # Filter valid coordinates with finite check
                    valid_x = (positions[:, 0] >= 0) & (positions[:, 0] < frame_width) & np.isfinite(positions[:, 0])
                    valid_y = (positions[:, 1] >= 0) & (positions[:, 1] < frame_height) & np.isfinite(positions[:, 1])
                    valid_mask = valid_x & valid_y
                    
                    if np.any(valid_mask):
                        valid_positions = positions[valid_mask]
                        valid_track_ids = np.array(classified_visible)[valid_mask]
                        
                        # Plot by level for efficiency - BIG POINTS
                        for level in range(len(outlier_tracks)):
                            level_mask = np.array([track_levels[tid] == level for tid in valid_track_ids])
                            if np.any(level_mask):
                                level_positions = valid_positions[level_mask]
                                
                                # Use scatter for speed with BIG POINTS and thick edges
                                ax.scatter(level_positions[:, 0], level_positions[:, 1], 
                                         c=colors[level], s=marker_size, alpha=0.9, 
                                         edgecolors='black', linewidths=2.0)
            
            # Convert to PIL with consistent settings
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                       pad_inches=0.1, facecolor='white', orientation='portrait')
            buf.seek(0)
            gif_frames.append(Image.open(buf).copy())
            buf.close()
            
            # Clear figure completely to prevent any rotation/scaling carryover
            plt.clf()
            plt.close(fig)
            plt.close('all')  # Ensure all figures are closed
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue
    
    # Save GIF with optimization
    if gif_frames:
        gif_frames[0].save(output_path, save_all=True, append_images=gif_frames[1:], 
                          duration=duration, loop=0, optimize=True, quality=95)
        print(f"Hierarchical outliers GIF saved: {output_path} ({len(gif_frames)} frames)")
        print(f"GIF size: {gif_frames[0].size}")
    else:
        print("No frames processed successfully")


def save_multi_frame_points_for_web_with_colored_tracks_and_error_lines(frames, model, device, output_path, 
                                   frame_indices=None, max_points_per_frame=20000, verbose=True, 
                                   pred_tracks=None, pred_visibility=None, track_labels=None,
                                   all_results=None):
    """
    Save 3D points from multiple frames for web visualization with colored track points and 3D error lines
    
    Args:
        frames: Array of video frames (T, H, W, 3)
        model: MoGe model for depth estimation
        device: Device to run inference on
        output_path: Path to save JSON file (e.g., "multi_frame_points.json")
        frame_indices: List of frame indices to process (None = all frames)
        max_points_per_frame: Maximum points per frame for performance
        verbose: Print progress information
        pred_tracks: Predicted tracks tensor (B, T, N, 2) - optional
        pred_visibility: Predicted visibility tensor (B, T, N) - optional
        track_labels: Cluster labels for tracks (shape: n_tracks,), -1 for unclustered/error tracks
        all_results: Results from pose estimation containing poses and camera matrices
    """
    import torch
    import numpy as np
    import json
    import cv2
    
    if frame_indices is None:
        frame_indices = list(range(len(frames)))
    
    # Extract poses from results if provided
    poses = None
    frame_pairs = None
    if all_results is not None:
        pose_results = all_results[('track_based', 'frame_zero_to_all')]
        poses = pose_results[0]  # List of (R, t) tuples
        frame_pair_details = pose_results[2]['frame_pair_details']
        frame_pairs = frame_pair_details['frame_pairs']
        
        if verbose:
            valid_poses = sum(1 for pose in poses if pose is not None and pose[0] is not None)
            print(f"Found {valid_poses}/{len(poses)} valid poses from pose estimation")
            print(f"Frame pairs: {frame_pairs[:5]}..." if len(frame_pairs) > 5 else f"Frame pairs: {frame_pairs}")
    
    # Generate colors for clusters
    def generate_cluster_colors(labels):
        """Generate distinct colors for each cluster label"""
        unique_labels = np.unique(labels)
        colors = {}
        
        # Use a color palette that cycles through distinct hues
        color_palette = [
            [1.0, 0.0, 0.0],    # Red
            [0.0, 1.0, 0.0],    # Green  
            [0.0, 0.0, 1.0],    # Blue
            [1.0, 1.0, 0.0],    # Yellow
            [1.0, 0.0, 1.0],    # Magenta
            [0.0, 1.0, 1.0],    # Cyan
            [1.0, 0.5, 0.0],    # Orange
            [0.5, 0.0, 1.0],    # Purple
            [0.0, 0.5, 0.0],    # Dark Green
            [0.5, 0.5, 0.0],    # Olive
            [0.0, 0.5, 0.5],    # Teal
            [0.5, 0.0, 0.5],    # Maroon
            [1.0, 0.75, 0.8],   # Pink
            [0.75, 0.75, 0.75], # Light Gray
            [0.5, 0.25, 0.0],   # Brown
        ]
        
        # Special color for error tracks (label -1)
        colors[-1] = [0.0, 0.0, 0.0]  # Black for unclustered/error tracks
        
        color_idx = 0
        for label in unique_labels:
            if label != -1:  # Skip -1 as it's already assigned
                colors[label] = color_palette[color_idx % len(color_palette)]
                color_idx += 1
        
        if verbose:
            print(f"Generated colors for {len(unique_labels)} cluster labels:")
            for label in unique_labels:
                color = colors[label]
                print(f"  Label {label}: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
        
        return colors
    
    # Generate cluster colors if labels provided
    cluster_colors = {}
    if track_labels is not None:
        cluster_colors = generate_cluster_colors(track_labels)
        if verbose:
            n_clusters = len(np.unique(track_labels[track_labels != -1]))
            n_error_tracks = np.sum(track_labels == -1)
            print(f"Track clustering info:")
            print(f"  Total tracks: {len(track_labels)}")
            print(f"  Valid clusters: {n_clusters}")
            print(f"  Error tracks (-1): {n_error_tracks}")
    
    all_frames_data = []
    global_bounds = {
        "min_x": float('inf'), "max_x": float('-inf'),
        "min_y": float('inf'), "max_y": float('-inf'), 
        "min_z": float('inf'), "max_z": float('-inf')
    }
    
    # Convert tracks to numpy if provided
    tracks_np = None
    visibility_np = None
    if pred_tracks is not None and pred_visibility is not None:
        tracks_np = pred_tracks[0].cpu().numpy()  # Shape: (T, N, 2)
        visibility_np = pred_visibility[0].cpu().numpy()  # Shape: (T, N)
        
        # Handle case where visibility might have extra dimension
        if len(visibility_np.shape) == 3 and visibility_np.shape[2] == 1:
            visibility_np = visibility_np.squeeze(-1)  # Remove last dimension if it's 1
        
        if verbose:
            print(f"Including {tracks_np.shape[1]} tracks with colored points in visualization")
            print(f"Tracks shape: {tracks_np.shape}")  
            print(f"Visibility shape: {visibility_np.shape}")
    
    # Pre-compute 3D coordinates for all frames to enable arrow and error line creation
    frame_3d_coords = {}
    if verbose:
        print(f"Pre-computing 3D coordinates for all frames...")
    
    for frame_idx in frame_indices:
        if verbose and frame_idx % 5 == 0:
            print(f"  Computing 3D coords for frame {frame_idx}")
        
        try:
            points, mask, intrinsics = get_3d_coordinates_for_frame(frames, model, device, frame_idx)
            frame_3d_coords[frame_idx] = {
                'points': points.cpu().numpy() if hasattr(points, 'cpu') else points,
                'mask': mask.cpu().numpy() if hasattr(mask, 'cpu') else mask,
                'intrinsics': intrinsics
            }
        except Exception as e:
            if verbose:
                print(f"  Error computing 3D coords for frame {frame_idx}: {e}")
            frame_3d_coords[frame_idx] = None
    
    # Get 3D coordinates for frame 0 (reference frame) for transformations
    frame0_3d_coords = {}
    if poses is not None and 0 in frame_3d_coords and frame_3d_coords[0] is not None:
        frame0_data = frame_3d_coords[0]
        frame0_points = frame0_data['points']
        frame0_mask = frame0_data['mask']
        
        # Extract 3D coordinates for all tracks in frame 0
        if tracks_np is not None and visibility_np is not None:
            for track_id in range(tracks_np.shape[1]):
                if visibility_np[0, track_id] > 0.5:  # Visible in frame 0
                    track_pos = tracks_np[0, track_id]
                    x, y = int(np.clip(track_pos[0], 0, frame0_points.shape[1] - 1)), \
                           int(np.clip(track_pos[1], 0, frame0_points.shape[0] - 1))
                    
                    if frame0_mask[y, x]:  # Valid depth
                        point_3d = frame0_points[y, x]
                        if np.isfinite(point_3d).all() and not np.allclose(point_3d, 0):
                            frame0_3d_coords[track_id] = point_3d
        
        if verbose:
            print(f"Extracted 3D coordinates for {len(frame0_3d_coords)} tracks from frame 0")
    
    if verbose:
        print(f"Processing {len(frame_indices)} frames with colored track points and error lines...")
    
    total_error_lines = 0
    total_predicted_points = 0
    
    for i, frame_idx in enumerate(frame_indices):
        if verbose and i % 5 == 0:
            print(f"Processing frame {frame_idx} ({i+1}/{len(frame_indices)})")
        
        try:
            frame_data = frame_3d_coords[frame_idx]
            if frame_data is None:
                continue
                
            points_np = frame_data['points']
            mask_np = frame_data['mask']
            
            # Get valid points for depth visualization - these will all be WHITE
            valid_mask = mask_np.astype(bool)
            depth_mask = (points_np[:, :, 2] > 0.1) & (points_np[:, :, 2] < 100.0)
            finite_mask = np.isfinite(points_np).all(axis=2)
            combined_mask = valid_mask & depth_mask & finite_mask
            
            valid_indices = np.where(combined_mask)
            
            if len(valid_indices[0]) == 0:
                if verbose:
                    print(f"  No valid depth points in frame {frame_idx}, skipping")
                continue
            
            # Extract valid points and use RGB colors from the original frame
            valid_points = points_np[valid_indices]
            
            # Get RGB colors from the original frame at the valid pixel locations
            frame_rgb = frames[frame_idx]  # Shape: (H, W, 3)
            valid_rows, valid_cols = valid_indices
            
            # Extract RGB colors at valid pixel locations and normalize to [0, 1]
            depth_colors = frame_rgb[valid_rows, valid_cols].astype(np.float32) / 255.0
            
            # Subsample depth points if too many
            if len(valid_points) > max_points_per_frame:
                indices = np.random.choice(len(valid_points), max_points_per_frame, replace=False)
                valid_points = valid_points[indices]
                depth_colors = depth_colors[indices]
            
            # Process track points with cluster colors
            track_arrows = []
            track_points = []
            track_point_colors = []
            
            # NEW: Process predicted points and error lines
            predicted_points = []
            error_lines = []
            
            if (tracks_np is not None and visibility_np is not None and 
                frame_idx < tracks_np.shape[0] and track_labels is not None):
                
                # Get visible tracks in current frame
                current_visible = visibility_np[frame_idx] > 0.5
                
                if np.any(current_visible):
                    # Get track positions for current frame
                    visible_track_indices = np.where(current_visible)[0]
                    current_track_positions = tracks_np[frame_idx, current_visible]  # Shape: (M, 2)
                    
                    # Convert to pixel coordinates
                    current_cols = np.clip(current_track_positions[:, 0].astype(int), 0, points_np.shape[1] - 1)
                    current_rows = np.clip(current_track_positions[:, 1].astype(int), 0, points_np.shape[0] - 1)
                    
                    # Check validity for current frame
                    current_valid = combined_mask[current_rows, current_cols]
                    
                    if np.any(current_valid):
                        # Extract 3D coordinates for valid track points
                        valid_current_rows = current_rows[current_valid]
                        valid_current_cols = current_cols[current_valid]
                        valid_track_indices = visible_track_indices[current_valid]
                        
                        current_3d = points_np[valid_current_rows, valid_current_cols]
                        
                        # Assign colors based on cluster labels for observed points
                        for j, track_idx in enumerate(valid_track_indices):
                            if track_idx < len(track_labels):  # Safety check
                                label = track_labels[track_idx]
                                color = cluster_colors.get(label, [0.5, 0.5, 0.5])  # Default gray if label not found
                                
                                track_points.append(current_3d[j].tolist())
                                track_point_colors.append(color)
                        
                        # NEW: Create predicted points and error lines using poses
                        if poses is not None and frame_pairs is not None and frame_idx > 0:
                            # Find the pose for this frame
                            pose_idx = None
                            for pair_idx, (ref_frame, target_frame) in enumerate(frame_pairs):
                                if target_frame == frame_idx:
                                    pose_idx = pair_idx
                                    break
                            
                            if (pose_idx is not None and pose_idx < len(poses) and 
                                poses[pose_idx] is not None and poses[pose_idx][0] is not None):
                                
                                R, t = poses[pose_idx]
                                
                                # Transform frame 0 points to current frame for tracks that are visible in both frames
                                for j, track_idx in enumerate(valid_track_indices):
                                    if track_idx in frame0_3d_coords:
                                        # Get 3D point from frame 0
                                        point_3d_frame0 = frame0_3d_coords[track_idx]
                                        
                                        # Transform: P_predicted = R * P_frame0 + t
                                        predicted_3d = R @ point_3d_frame0 + t
                                        
                                        # Check if prediction is reasonable
                                        if (np.isfinite(predicted_3d).all() and 
                                            0.1 < predicted_3d[2] < 100.0):  # Reasonable depth
                                            
                                            predicted_points.append(predicted_3d.tolist())
                                            
                                            # Create error line from observed to predicted
                                            observed_3d = current_3d[j]
                                            
                                            # Get the cluster color for this track
                                            if track_idx < len(track_labels):
                                                label = track_labels[track_idx]
                                                line_color = cluster_colors.get(label, [1.0, 1.0, 1.0])  # Default white if label not found
                                            else:
                                                line_color = [1.0, 1.0, 1.0]  # White for tracks without labels
                                            
                                            error_lines.append({
                                                'observed': observed_3d.tolist(),
                                                'predicted': predicted_3d.tolist(),
                                                'error': float(np.linalg.norm(predicted_3d - observed_3d)),
                                                'track_id': int(track_idx),
                                                'color': line_color  # Add color information
                                            })
                
                # Create arrows for next frame (keeping original arrow logic)
                next_frame_idx = frame_idx + 1
                if next_frame_idx in frame_3d_coords and frame_3d_coords[next_frame_idx] is not None:
                    next_frame_data = frame_3d_coords[next_frame_idx]
                    next_points_np = next_frame_data['points']
                    next_mask_np = next_frame_data['mask']
                    next_combined_mask = (next_mask_np.astype(bool) & 
                                        (next_points_np[:, :, 2] > 0.1) & 
                                        (next_points_np[:, :, 2] < 100.0) & 
                                        np.isfinite(next_points_np).all(axis=2))
                    
                    # Find tracks visible in both current and next frame
                    current_visible = visibility_np[frame_idx] > 0.5
                    next_visible = visibility_np[next_frame_idx] > 0.5 if next_frame_idx < visibility_np.shape[0] else np.zeros_like(current_visible)
                    both_visible = current_visible & next_visible
                    
                    if np.any(both_visible):
                        # Get track positions for both frames
                        current_track_positions = tracks_np[frame_idx, both_visible]
                        next_track_positions = tracks_np[next_frame_idx, both_visible]
                        
                        # Convert to pixel coordinates for both frames
                        current_cols = np.clip(current_track_positions[:, 0].astype(int), 0, points_np.shape[1] - 1)
                        current_rows = np.clip(current_track_positions[:, 1].astype(int), 0, points_np.shape[0] - 1)
                        next_cols = np.clip(next_track_positions[:, 0].astype(int), 0, next_points_np.shape[1] - 1)
                        next_rows = np.clip(next_track_positions[:, 1].astype(int), 0, next_points_np.shape[0] - 1)
                        
                        # Check validity for both frames
                        current_valid = combined_mask[current_rows, current_cols]
                        next_valid = next_combined_mask[next_rows, next_cols]
                        arrow_valid = current_valid & next_valid
                        
                        if np.any(arrow_valid):
                            # Extract 3D coordinates for valid arrows
                            valid_current_rows = current_rows[arrow_valid]
                            valid_current_cols = current_cols[arrow_valid]
                            valid_next_rows = next_rows[arrow_valid]
                            valid_next_cols = next_cols[arrow_valid]
                            
                            current_3d = points_np[valid_current_rows, valid_current_cols]
                            next_3d = next_points_np[valid_next_rows, valid_next_cols]
                            
                            # Additional filtering for good arrows
                            arrow_distance = np.linalg.norm(next_3d - current_3d, axis=1)
                            distance_filter = (arrow_distance > 0.01) & (arrow_distance < 5.0)
                            
                            if np.any(distance_filter):
                                final_current_3d = current_3d[distance_filter]
                                final_next_3d = next_3d[distance_filter]
                                
                                # Create arrow data
                                for j in range(len(final_current_3d)):
                                    track_arrows.append({
                                        'start': final_current_3d[j].tolist(),
                                        'end': final_next_3d[j].tolist(),
                                        'length': float(np.linalg.norm(final_next_3d[j] - final_current_3d[j]))
                                    })
            
            total_error_lines += len(error_lines)
            total_predicted_points += len(predicted_points)
            
            if verbose and frame_idx == frame_indices[0]:
                print(f"Frame {frame_idx} final summary:")
                print(f"  Depth points (white): {len(valid_points)}")  
                print(f"  Track points (colored): {len(track_points)}")
                print(f"  Track arrows (red): {len(track_arrows)}")
                print(f"  Predicted points (cyan): {len(predicted_points)}")
                print(f"  Error lines: {len(error_lines)}")
            
            # Combine all points and colors
            all_points = valid_points.tolist() + track_points
            all_colors = depth_colors.tolist() + track_point_colors
            
            # Update global bounds (include predicted points in bounds calculation)
            all_points_for_bounds = all_points + predicted_points
            if all_points_for_bounds:
                combined_points = np.array(all_points_for_bounds)
                global_bounds["min_x"] = min(global_bounds["min_x"], float(combined_points[:, 0].min()))
                global_bounds["max_x"] = max(global_bounds["max_x"], float(combined_points[:, 0].max()))
                global_bounds["min_y"] = min(global_bounds["min_y"], float(combined_points[:, 1].min()))
                global_bounds["max_y"] = max(global_bounds["max_y"], float(combined_points[:, 1].max()))
                global_bounds["min_z"] = min(global_bounds["min_z"], float(combined_points[:, 2].min()))
                global_bounds["max_z"] = max(global_bounds["max_z"], float(combined_points[:, 2].max()))
            
            # Store frame data
            frame_data = {
                "frame_index": frame_idx,
                "points": all_points,
                "colors": all_colors,
                "count": len(all_points),
                "depth_point_count": len(valid_points),
                "track_point_count": len(track_points),
                "track_arrows": track_arrows,
                "arrow_count": len(track_arrows),
                "predicted_points": predicted_points,
                "predicted_point_count": len(predicted_points),
                "error_lines": error_lines,
                "error_line_count": len(error_lines)
            }
            
            all_frames_data.append(frame_data)
            
            if verbose and len(all_frames_data) % 10 == 0:
                print(f"  Processed {len(all_frames_data)} frames so far...")
                
        except Exception as e:
            if verbose:
                print(f"  Error processing frame {frame_idx}: {e}")
            continue
    
    if not all_frames_data:
        print("No valid frames processed!")
        return
    
    # Prepare final data structure
    total_points = sum(frame["count"] for frame in all_frames_data)
    total_depth_points = sum(frame["depth_point_count"] for frame in all_frames_data)
    total_track_points = sum(frame["track_point_count"] for frame in all_frames_data)
    total_arrows = sum(frame["arrow_count"] for frame in all_frames_data)
    
    # Prepare final data structure - keep original type for compatibility
    multi_frame_data = {
        "type": "multi_frame_point_cloud_with_arrows",  # Keep original type
        "total_frames": len(all_frames_data),
        "frames": all_frames_data,
        "global_bounds": global_bounds,
        "total_points": total_points,
        "total_arrows": total_arrows,
        "total_error_lines": total_error_lines,
        "total_predicted_points": total_predicted_points,
        "frame_indices": [frame["frame_index"] for frame in all_frames_data],
        "has_arrows": tracks_np is not None,
        "has_error_lines": poses is not None and total_error_lines > 0,
        "has_predicted_points": poses is not None and total_predicted_points > 0,
        # Additional metadata for colored tracks
        "has_colored_tracks": track_labels is not None,
        "total_depth_points": total_depth_points,
        "total_track_points": total_track_points,
        "cluster_info": {
            "unique_labels": [int(label) for label in np.unique(track_labels)] if track_labels is not None else [],
            "cluster_colors": {int(k): v for k, v in cluster_colors.items()}
        }
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(multi_frame_data, f, indent=2)
    
    if verbose:
        print(f"\nSaved multi-frame point cloud with colored tracks and error lines to {output_path}")
        print(f"Total frames: {len(all_frames_data)}")
        print(f"Total depth points (white): {total_depth_points:,}")
        print(f"Total track points (colored): {total_track_points:,}")
        if tracks_np is not None:
            print(f"Total track arrows: {total_arrows:,}")
        if poses is not None:
            print(f"Total predicted points (cyan): {total_predicted_points:,}")
            print(f"Total error lines: {total_error_lines:,}")
        if track_labels is not None:
            n_clusters = len(np.unique(track_labels[track_labels != -1]))
            n_error_tracks = np.sum(track_labels == -1)
            print(f"Cluster summary: {n_clusters} valid clusters, {n_error_tracks} error tracks")
        print(f"Global bounds:")
        print(f"  X[{global_bounds['min_x']:.2f}, {global_bounds['max_x']:.2f}]")
        print(f"  Y[{global_bounds['min_y']:.2f}, {global_bounds['max_y']:.2f}]")
        print(f"  Z[{global_bounds['min_z']:.2f}, {global_bounds['max_z']:.2f}]")

# Usage example:
# save_multi_frame_points_for_web_with_colored_tracks_and_error_lines(
#     frames, model, 'cuda', 
#     path + f"/multi_frame_points_with_error_lines_{vid}.json",
#     frame_indices=frame_indices,
#     max_points_per_frame=15000,
#     verbose=True,
#     pred_tracks=pred_tracks, 
#     pred_visibility=pred_visibility,
#     track_labels=labels,  # Your cluster labels array
#     all_results=all_results  # Results from first iteration containing poses
# )

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2

def create_reprojection_error_visualization(frames, pred_tracks, pred_visibility, outlier_tracks, 
                                          all_results, model, device, output_path, 
                                          duration=600, figsize=(12, 8), error_threshold=10.0, 
                                          show_error_lines=True, show_inliers=True, show_outliers=True):
    """
    Create a GIF visualization showing reprojection errors for tracks.
    
    Args:
        frames: Video frames (numpy array)
        pred_tracks: Predicted track positions 
        pred_visibility: Track visibility data
        outlier_tracks: List of outlier track sets (we'll use outlier_tracks[0])
        all_results: Results from pose estimation containing poses and camera matrices
        model: MoGe model for getting 3D coordinates
        device: Device for computation
        output_path: Path to save the GIF
        duration: Duration per frame in ms
        figsize: Figure size
        error_threshold: Maximum error to display (for color scaling)
        show_error_lines: Whether to draw lines showing reprojection errors
        show_inliers: Whether to show inlier tracks
        show_outliers: Whether to show outlier tracks
    """
    
    # Extract data from results
    pose_results = all_results[('track_based', 'frame_zero_to_all')]
    poses = pose_results[0]  # List of (R, t) tuples
    frame_pair_details = pose_results[2]['frame_pair_details']
    
    # Camera matrices might not be stored, so we need to compute them
    # Get camera matrices by computing intrinsics for each frame
    camera_matrices = {}
    H, W = frames[0].shape[:2]
    
    print("Computing camera matrices for each frame...")
    for frame_idx in range(len(frames)):
        _, _, intrinsics = get_3d_coordinates_for_frame(frames, model, device, frame_idx)
        camera_matrices[frame_idx] = convert_moge_intrinsics_to_opencv(intrinsics, H, W)
    
    print(f"Computed camera matrices for {len(camera_matrices)} frames")
    
    # Get frame pairs (should be (0, 1), (0, 2), ..., (0, T-1))
    frame_pairs = frame_pair_details['frame_pairs']
    
    print(f"Debug info:")
    print(f"  Number of poses: {len(poses)}")
    print(f"  Number of frame pairs: {len(frame_pairs)}")
    print(f"  First few frame pairs: {frame_pairs[:5] if len(frame_pairs) > 5 else frame_pairs}")
    print(f"  First few poses valid: {[(i, poses[i] is not None and poses[i][0] is not None) if i < len(poses) else (i, 'Out of range') for i in range(min(5, len(frame_pairs)))]}")
    
    # Get track sets
    outlier_track_set = outlier_tracks[0] if outlier_tracks else set()
    inlier_track_set = frame_pair_details.get('global_inlier_tracks', set())
    
    print(f"  Outlier tracks: {len(outlier_track_set)}")
    print(f"  Inlier tracks: {len(inlier_track_set)}")
    
    # Get 3D coordinates for frame 0 (reference frame)
    points_3d_frame0, mask_frame0, intrinsics_frame0 = get_3d_coordinates_for_frame(
        frames, model, device, 0
    )
    points_3d_frame0 = points_3d_frame0.cpu().numpy()
    mask_frame0 = mask_frame0.cpu().numpy()
    
    # Convert tracks and visibility to numpy
    tracks_np = pred_tracks[0].cpu().numpy()  # Shape: (T, N, 2)
    visibility_np = pred_visibility[0].cpu().numpy()  # Shape: (T, N)
    
    T, N = tracks_np.shape[:2]
    
    # Precompute 3D coordinates for all tracks in frame 0
    track_3d_coords = {}
    for track_id in range(N):
        if visibility_np[0, track_id] > 0.5:  # Visible in frame 0
            track_pos = tracks_np[0, track_id]
            x, y = int(np.clip(track_pos[0], 0, points_3d_frame0.shape[1] - 1)), \
                   int(np.clip(track_pos[1], 0, points_3d_frame0.shape[0] - 1))
            
            if mask_frame0[y, x]:  # Valid depth
                point_3d = points_3d_frame0[y, x]
                if np.isfinite(point_3d).all() and not np.allclose(point_3d, 0):
                    track_3d_coords[track_id] = point_3d
    
    print(f"Found 3D coordinates for {len(track_3d_coords)} tracks in frame 0")
    
    # Setup the plot with tight layout to reduce padding
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    
    def animate(frame_idx):
        ax.clear()
        
        # Display the current frame
        ax.imshow(frames[frame_idx])
        ax.set_title(f'Frame {frame_idx}', fontsize=40)
        ax.axis('off')
        
        if frame_idx == 0:
            # In frame 0, just show the tracks without reprojection
            for track_id in range(N):
                if visibility_np[frame_idx, track_id] > 0.5:
                    track_pos = tracks_np[frame_idx, track_id]
                    
                    if track_id in outlier_track_set and show_outliers:
                        ax.plot(track_pos[0], track_pos[1], 'ro', markersize=4, label='Outlier' if track_id == list(outlier_track_set)[0] else "")
                    elif track_id in inlier_track_set and show_inliers:
                        ax.plot(track_pos[0], track_pos[1], 'go', markersize=4, label='Inlier' if track_id == list(inlier_track_set)[0] else "")
            return
        
        # Find the corresponding pose for this frame
        pose_idx = None
        for i, (ref_frame, target_frame) in enumerate(frame_pairs):
            if target_frame == frame_idx:
                pose_idx = i
                break
        
        if pose_idx is None or pose_idx >= len(poses) or poses[pose_idx] is None or poses[pose_idx][0] is None:
            # No pose available for this frame
            ax.text(0.5, 0.5, f'No pose available for frame {frame_idx}', 
                   transform=ax.transAxes, ha='center', va='center', 
                   fontsize=14, color='red', weight='bold')
            return
        
        R, t = poses[pose_idx]
        camera_matrix = camera_matrices[frame_idx]
        
        # Project 3D points and compute reprojection errors
        reprojection_errors = {}
        projected_points = {}
        
        for track_id, point_3d in track_3d_coords.items():
            if visibility_np[frame_idx, track_id] > 0.5:  # Visible in current frame
                # Transform 3D point: P_t = R * P_0 + t
                transformed_point = R @ point_3d + t
                
                # Project to 2D
                try:
                    rvec, _ = cv2.Rodrigues(R)
                    tvec = t.reshape(3, 1)
                    
                    projected_2d, _ = cv2.projectPoints(
                        point_3d.reshape(1, 1, 3),
                        rvec,
                        tvec,
                        camera_matrix,
                        np.zeros(4, dtype=np.float32)
                    )
                    projected_2d = projected_2d.reshape(2)
                    
                    # Get observed 2D point
                    observed_2d = tracks_np[frame_idx, track_id]
                    
                    # Compute reprojection error
                    error = np.linalg.norm(projected_2d - observed_2d)
                    
                    reprojection_errors[track_id] = error
                    projected_points[track_id] = projected_2d
                    
                except Exception as e:
                    continue
        
        # Visualize tracks with reprojection errors
        max_error = max(reprojection_errors.values()) if reprojection_errors else 1.0
        error_scale = min(error_threshold, max_error)
        
        for track_id in range(N):
            if (visibility_np[frame_idx, track_id] > 0.5 and 
                track_id in reprojection_errors):
                
                observed_pos = tracks_np[frame_idx, track_id]
                projected_pos = projected_points[track_id]
                error = reprojection_errors[track_id]
                
                # Color based on error magnitude
                error_normalized = min(error / error_scale, 1.0)
                
                # Show tracks based on inlier/outlier status
                if track_id in outlier_track_set and show_outliers:
                    # Red for outliers, intensity based on error
                    color = (1.0, 1.0 - error_normalized, 1.0 - error_normalized)
                    ax.plot(observed_pos[0], observed_pos[1], 'o', color=color, 
                           markersize=5, markeredgecolor='red', markeredgewidth=1)
                    
                    # Show projected point
                    ax.plot(projected_pos[0], projected_pos[1], 's', color=color, 
                           markersize=4, markeredgecolor='darkred', markeredgewidth=1)
                    
                elif track_id in inlier_track_set and show_inliers:
                    # Green for inliers, intensity based on error
                    color = (1.0 - error_normalized, 1.0, 1.0 - error_normalized)
                    ax.plot(observed_pos[0], observed_pos[1], 'o', color=color, 
                           markersize=5, markeredgecolor='green', markeredgewidth=1)
                    
                    # Show projected point
                    ax.plot(projected_pos[0], projected_pos[1], 's', color=color, 
                           markersize=4, markeredgecolor='darkgreen', markeredgewidth=1)
                
                # Draw error line
                if show_error_lines and error > 0.5:  # Only show lines for non-trivial errors
                    line_alpha = min(0.8, error / error_scale)
                    line_color = 'red' if track_id in outlier_track_set else 'yellow'
                    ax.plot([observed_pos[0], projected_pos[0]], 
                           [observed_pos[1], projected_pos[1]], 
                           color=line_color, alpha=line_alpha, linewidth=1)
        
        # # Add legend and error statistics
        # legend_elements = []
        # if show_inliers and inlier_track_set:
        #     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
        #                                     markerfacecolor='green', markersize=8, 
        #                                     label='Inlier (observed)'))
        #     legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
        #                                     markerfacecolor='green', markersize=6, 
        #                                     label='Inlier (projected)'))
        # if show_outliers and outlier_track_set:
        #     legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
        #                                     markerfacecolor='red', markersize=8, 
        #                                     label='Outlier (observed)'))
        #     legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
        #                                     markerfacecolor='red', markersize=6, 
        #                                     label='Outlier (projected)'))
        
        # if legend_elements:
        #     ax.legend(handles=legend_elements, loc='upper right')
        
        # # Add error statistics text
        # if reprojection_errors:
        #     mean_error = np.mean(list(reprojection_errors.values()))
        #     max_error = np.max(list(reprojection_errors.values()))
        #     ax.text(0.02, 0.98, f'Mean Error: {mean_error:.2f}px\nMax Error: {max_error:.2f}px\nTracks: {len(reprojection_errors)}', 
        #            transform=ax.transAxes, va='top', ha='left', 
        #            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        #            fontsize=10)
    
    # Create animation
    print(f"Creating reprojection error visualization for {T} frames...")
    anim = FuncAnimation(fig, animate, frames=T, interval=duration, repeat=True)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=1000/duration)
    plt.close(fig)
    
    print(f"Reprojection error visualization saved to: {output_path}")

# Usage example (add this to your script):
# create_reprojection_error_visualization(
#     frames=frames,
#     pred_tracks=pred_tracks,
#     pred_visibility=pred_visibility,
#     outlier_tracks=outlier_tracks,
#     all_results=all_results,
#     model=model,
#     device=device,
#     output_path=path + f'/reprojection_errors_{vid}.gif',
#     duration=600,
#     error_threshold=10.0,
#     show_error_lines=True,
#     show_inliers=True,
#     show_outliers=True
# )

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2

def create_multi_iteration_reprojection_visualization(frames, pred_tracks, pred_visibility, 
                                                    outlier_tracks, all_results_list, 
                                                    model, device, output_path, 
                                                    duration=600, figsize=(12, 8), 
                                                    error_threshold=10.0, show_error_lines=True):
    """
    Create a GIF visualization showing reprojection errors for tracks across multiple RANSAC iterations.
    
    For each track, uses the pose from the iteration where it first became an inlier:
    - If outlier in iteration 0 but inlier in iteration 1 -> use iteration 0 poses
    - If outlier in iteration 1 but inlier in iteration 2 -> use iteration 1 poses
    - etc.
    
    Args:
        frames: Video frames (numpy array)
        pred_tracks: Predicted track positions 
        pred_visibility: Track visibility data
        outlier_tracks: List of outlier track sets from each iteration
        all_results_list: List of all_results from each iteration
        model: MoGe model for getting 3D coordinates
        device: Device for computation
        output_path: Path to save the GIF
        duration: Duration per frame in ms
        figsize: Figure size
        error_threshold: Maximum error to display (for color scaling)
        show_error_lines: Whether to draw lines showing reprojection errors
    """
    
    # Extract poses and frame pairs from each iteration
    iterations_data = []
    for i, all_results in enumerate(all_results_list):
        pose_results = all_results[('track_based', 'frame_zero_to_all')]
        poses = pose_results[0]
        frame_pair_details = pose_results[2]['frame_pair_details']
        frame_pairs = frame_pair_details['frame_pairs']
        
        iterations_data.append({
            'poses': poses,
            'frame_pairs': frame_pairs,
            'outlier_tracks': outlier_tracks[i],
            'inlier_tracks': frame_pair_details.get('global_inlier_tracks', set())
        })
        
        print(f"Iteration {i}: {len(outlier_tracks[i])} outliers, {len(frame_pair_details.get('global_inlier_tracks', set()))} inliers")
    
    # Determine which iteration's pose to use for each track
    track_iteration_mapping = {}
    all_track_ids = set()
    
    # Collect all track IDs
    for iteration_data in iterations_data:
        all_track_ids.update(iteration_data['outlier_tracks'])
        all_track_ids.update(iteration_data['inlier_tracks'])
    
    print(f"Total unique tracks across all iterations: {len(all_track_ids)}")
    
    # For each track, find the first iteration where it became an inlier
    for track_id in all_track_ids:
        track_iteration_mapping[track_id] = None
        
        for i, iteration_data in enumerate(iterations_data):
            if track_id in iteration_data['inlier_tracks']:
                # This track became an inlier in iteration i
                # Use poses from the previous iteration (i-1)
                # If this is iteration 0, still use iteration 0 poses
                pose_iteration = i - 1 if i > 0 else 0
                track_iteration_mapping[track_id] = pose_iteration
                break
        
        # If track was never an inlier, use the last iteration
        if track_iteration_mapping[track_id] is None:
            track_iteration_mapping[track_id] = len(iterations_data) - 1
    
    # Debug: show mapping summary
    mapping_summary = {}
    for track_id, iteration in track_iteration_mapping.items():
        if iteration not in mapping_summary:
            mapping_summary[iteration] = 0
        mapping_summary[iteration] += 1
    
    print("Track-to-iteration mapping summary:")
    for iteration, count in sorted(mapping_summary.items()):
        print(f"  Iteration {iteration}: {count} tracks")
    
    # Compute camera matrices for first 6 frames only
    camera_matrices = {}
    H, W = frames[0].shape[:2]
    frames_to_process = min(6, len(frames))
    
    print(f"Computing camera matrices for first {frames_to_process} frames...")
    for frame_idx in range(frames_to_process):
        _, _, intrinsics = get_3d_coordinates_for_frame(frames, model, device, frame_idx)
        camera_matrices[frame_idx] = convert_moge_intrinsics_to_opencv(intrinsics, H, W)
    
    # Get 3D coordinates for frame 0 (reference frame)
    points_3d_frame0, mask_frame0, intrinsics_frame0 = get_3d_coordinates_for_frame(
        frames, model, device, 0
    )
    points_3d_frame0 = points_3d_frame0.cpu().numpy()
    mask_frame0 = mask_frame0.cpu().numpy()
    
    # Convert tracks and visibility to numpy
    tracks_np = pred_tracks[0].cpu().numpy()  # Shape: (T, N, 2)
    visibility_np = pred_visibility[0].cpu().numpy()  # Shape: (T, N)
    
    T, N = tracks_np.shape[:2]
    
    # Precompute 3D coordinates for all tracks in frame 0
    track_3d_coords = {}
    for track_id in range(N):
        if visibility_np[0, track_id] > 0.5:  # Visible in frame 0
            track_pos = tracks_np[0, track_id]
            x, y = int(np.clip(track_pos[0], 0, points_3d_frame0.shape[1] - 1)), \
                   int(np.clip(track_pos[1], 0, points_3d_frame0.shape[0] - 1))
            
            if mask_frame0[y, x]:  # Valid depth
                point_3d = points_3d_frame0[y, x]
                if np.isfinite(point_3d).all() and not np.allclose(point_3d, 0):
                    track_3d_coords[track_id] = point_3d
    
    print(f"Found 3D coordinates for {len(track_3d_coords)} tracks in frame 0")
    
    # Setup the plot with tight layout to reduce padding
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    
    def animate(frame_idx):
        ax.clear()
        
        # Display the current frame
        ax.imshow(frames[frame_idx])
        ax.set_title(f'Frame {frame_idx}', fontsize=24, fontweight='bold')
        ax.axis('off')
        
        if frame_idx == 0:
            # In frame 0, show tracks colored by their final classification
            for track_id in range(N):
                if visibility_np[frame_idx, track_id] > 0.5:
                    track_pos = tracks_np[frame_idx, track_id]
                    
                    # Determine color based on final iteration classification
                    if track_id in iterations_data[-1]['outlier_tracks']:
                        ax.plot(track_pos[0], track_pos[1], 'ro', markersize=4)
                    elif track_id in iterations_data[-1]['inlier_tracks']:
                        ax.plot(track_pos[0], track_pos[1], 'go', markersize=4)
                    else:
                        ax.plot(track_pos[0], track_pos[1], 'ko', markersize=4)
            return
        
        # For other frames, show reprojection using appropriate iteration poses
        reprojection_errors = {}
        projected_points = {}
        
        for track_id, point_3d in track_3d_coords.items():
            if visibility_np[frame_idx, track_id] > 0.5:  # Visible in current frame
                
                # Get the iteration to use for this track
                iteration_to_use = track_iteration_mapping.get(track_id, 0)
                iteration_data = iterations_data[iteration_to_use]
                
                # Find the pose for this frame in the chosen iteration
                pose_idx = None
                for i, (ref_frame, target_frame) in enumerate(iteration_data['frame_pairs']):
                    if target_frame == frame_idx:
                        pose_idx = i
                        break
                
                if (pose_idx is None or pose_idx >= len(iteration_data['poses']) or 
                    iteration_data['poses'][pose_idx] is None or 
                    iteration_data['poses'][pose_idx][0] is None):
                    continue
                
                R, t = iteration_data['poses'][pose_idx]
                camera_matrix = camera_matrices[frame_idx]
                
                # Project 3D point using the chosen iteration's pose
                try:
                    rvec, _ = cv2.Rodrigues(R)
                    tvec = t.reshape(3, 1)
                    
                    projected_2d, _ = cv2.projectPoints(
                        point_3d.reshape(1, 1, 3),
                        rvec,
                        tvec,
                        camera_matrix,
                        np.zeros(4, dtype=np.float32)
                    )
                    projected_2d = projected_2d.reshape(2)
                    
                    # Get observed 2D point
                    observed_2d = tracks_np[frame_idx, track_id]
                    
                    # Compute reprojection error
                    error = np.linalg.norm(projected_2d - observed_2d)
                    
                    reprojection_errors[track_id] = error
                    projected_points[track_id] = projected_2d
                    
                except Exception as e:
                    continue
        
        # Visualize tracks with reprojection errors
        max_error = max(reprojection_errors.values()) if reprojection_errors else 1.0
        error_scale = min(error_threshold, max_error)
        
        # Color scheme: different colors for tracks from different iterations
        iteration_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
        
        for track_id in range(N):
            if (visibility_np[frame_idx, track_id] > 0.5 and 
                track_id in reprojection_errors):
                
                observed_pos = tracks_np[frame_idx, track_id]
                projected_pos = projected_points[track_id]
                error = reprojection_errors[track_id]
                
                # Get iteration and color
                iteration_used = track_iteration_mapping.get(track_id, 0)
                base_color = iteration_colors[iteration_used % len(iteration_colors)]
                
                # Color intensity based on error magnitude
                error_normalized = min(error / error_scale, 1.0)
                
                # Show observed point
                ax.plot(observed_pos[0], observed_pos[1], 'o', color=base_color, 
                       markersize=5, alpha=0.8, markeredgecolor='black', markeredgewidth=1)
                
                # Show projected point
                ax.plot(projected_pos[0], projected_pos[1], 's', color=base_color, 
                       markersize=4, alpha=0.8, markeredgecolor='black', markeredgewidth=1)
                
                # Draw error line
                if show_error_lines and error > 0.5:
                    line_alpha = min(0.8, error / error_scale)
                    ax.plot([observed_pos[0], projected_pos[0]], 
                           [observed_pos[1], projected_pos[1]], 
                           color=base_color, alpha=line_alpha, linewidth=1)
    
    # Create animation - only for first 6 frames
    frames_to_animate = min(6, T)
    print(f"Creating multi-iteration reprojection visualization for first {frames_to_animate} frames...")
    anim = FuncAnimation(fig, animate, frames=frames_to_animate, interval=duration, repeat=True)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=1000/duration)
    plt.close(fig)
    
    print(f"Multi-iteration reprojection visualization saved to: {output_path}")

# Usage example:
# create_multi_iteration_reprojection_visualization(
#     frames=frames,
#     pred_tracks=pred_tracks,
#     pred_visibility=pred_visibility,
#     outlier_tracks=outlier_tracks,  # Now expects a list of outlier sets
#     all_results_list=[all_results, updated_all_results],  # List of results from each iteration
#     model=model,
#     device=device,
#     output_path=path + f'/multi_iteration_reprojection_{vid}.gif',
#     duration=600,
#     error_threshold=10.0,
#     show_error_lines=True
# )