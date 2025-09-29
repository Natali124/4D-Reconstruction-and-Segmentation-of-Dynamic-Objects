import numpy as np
import torch

codes = {
    'parrots': '0a7b27fde9',
    'rabbit': '0b6f9105fc',
    # 'bull': '0b7dbfa3cb',
    # 'boat_and_dolphin': '0b9cea51ca',
    'camels': '0b9d012be8',
    'duck_and_fish': '0c3a04798c',
    'deer': '0c11c6bf7b',
    'man_and_tiger': '0c44a9d545',
    'zebra': '0d2fcc0dcd',
    'cats': '0dbaf284f1',
    'firetruck': '0e9ebe4e3c',
    'dolphins': '0f2ab8b1ff',
    'ducks_and_people': '0fa7b59356',
    'bus_forward': '1a5fe06b00',
    'cat_and_fox_fighting': '09ff54fef4',
    'scateboard_tricks': '11a6ba8c94',
    'space_oddysey_monkeys': '011ac0a06f',
    'turtles_swimming': '11c722a456',
    'guy_and_frisbee': '12bddb2bcb',
    'train_station_one': '17c7bcd146',
    'penguins': '18b245ab49',
    'pets_moving_in_a_line_looks_good': '21f4019116',
    'monkeys_on_a_bicycle': '28a8eb9623',
    'car_moving_super_fast': '39d584b09f',
    'fish_in_clear': '40f4026bf5',
    'cyclists_coming_forward': '44b4dad8c9',
    'bull_running': '44f4f57099',
    'sheep_coming_towards_me': '45bf0e947d',
    'also_sheep': '46e18e42f1',
    'shoop': '46f5093c59',
    'motorcycle': '53af427bb2',
    'train': '63d90c2bae'
}

def get_3d_coordinates_for_frame(frames, model, device, frame_idx):
    """Get 3D coordinates and intrinsics for a specific frame using MoGe"""
    input_image = frames[frame_idx]
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    output = model.infer(input_image)
    return output["points"], output["mask"], output["intrinsics"]

def create_labels_from_outlier_tracks(outlier_tracks, pred_tracks_shape):
    if isinstance(pred_tracks_shape, tuple):
        num_tracks = pred_tracks_shape[-2]
    else:
        num_tracks = pred_tracks_shape
    
    labels = np.full(num_tracks, -1, dtype=int)
    
    for level, outlier_set in enumerate(outlier_tracks):
        label_value = level + 1  # Labels start from 1
        for track_id in outlier_set:
            if track_id < num_tracks:  # Safety check
                labels[track_id] = label_value
    
    return labels

def compute_error(results):
        return np.mean(list((results[('track_based', 'frame_zero_to_all')])[2]['frame_pair_details']['global_track_errors'].values()))

def create_modified_visibility(pred_visibility, all_results, verbose=True):
    """
    Create modified visibility where only outlier tracks are visible.
    
    Args:
        pred_visibility: Original visibility tensor/array of shape (1, T, N) or (T, N)
        all_results: Dictionary containing track-based RANSAC results
        
    Returns:
        new_pred_visibility: Modified visibility where:
            - True only for outlier tracks (tracks not in inlier_tracks)
            - False for all inlier tracks
            - Maintains False where original visibility was False
    """
    # Make a copy to avoid modifying the original
    new_pred_visibility = pred_visibility.clone() if hasattr(pred_visibility, 'clone') else pred_visibility.copy()
    R, t, track_results = all_results[('track_based', 'frame_zero_to_all')]
    
    inlier_tracks = track_results['frame_pair_details']['global_inlier_tracks']
    
    # Handle different tensor/array shapes
    if len(new_pred_visibility.shape) == 3:  # Shape: (1, T, N)
        batch_dim = True
        visibility_data = new_pred_visibility[0]  # Work with (T, N)
    elif len(new_pred_visibility.shape) == 2:  # Shape: (T, N)
        batch_dim = False
        visibility_data = new_pred_visibility
    else:
        raise ValueError(f"Unexpected visibility shape: {new_pred_visibility.shape}")
    
    # Get total number of tracks
    num_tracks = visibility_data.shape[1]
    
    # Set visibility to False for all inlier tracks (across all frames)
    for track_id in inlier_tracks:
        if track_id < num_tracks:  # Ensure track_id is valid
            visibility_data[:, track_id] = False
    
    # Update the original tensor/array structure
    if batch_dim:
        new_pred_visibility[0] = visibility_data
    else:
        new_pred_visibility = visibility_data
    
    if verbose:
        # Print statistics
        if hasattr(pred_visibility, 'sum'):  # For tensors
            original_visible = pred_visibility.sum().item()
            new_visible = new_pred_visibility.sum().item()
        else:  # For numpy arrays
            original_visible = np.sum(pred_visibility)
            new_visible = np.sum(new_pred_visibility)
        
        print(f"Modified visibility:")
        print(f"  Original visible track-frame pairs: {original_visible}")
        print(f"  New visible track-frame pairs (outliers only): {new_visible}")
        print(f"  Number of inlier tracks hidden: {len(inlier_tracks)}")
        
    return new_pred_visibility

def create_inliers_outliers_gif(frames, pred_tracks, pred_visibility, all_results, 
                               output_path='inliers_outliers.gif', duration=300, figsize=(12, 8)):
    """
    Create a GIF showing inliers vs outliers across all frames
    Updated for frame-to-frame pose estimation results
    
    Args:
        frames: Array of video frames
        pred_tracks: Predicted tracks tensor
        pred_visibility: Predicted visibility tensor  
        all_results: Dictionary with results (now frame-to-frame based)
        output_path: Output GIF file path
        duration: Duration per frame in milliseconds
        figsize: Figure size for each frame
    """
    
    # Check if inputs are None
    if frames is None:
        print("Error: frames is None")
        return
    if pred_tracks is None:
        print("Error: pred_tracks is None")
        return
    if pred_visibility is None:
        print("Error: pred_visibility is None")
        return
    if all_results is None:
        print("Error: all_results is None")
        return
    
    k = next(iter(all_results))
    
    # Handle the new frame-to-frame results structure
    if k in all_results:
        pose_results, weighted_error, track_results = all_results[k]
        
        if track_results is None:
            print("Error: track_results is None")
            return
            
        # Extract information from the new structure
        track_correspondences = track_results.get('track_correspondences', {})
        frame_pair_details = track_results.get('frame_pair_details', {})
        
        if not track_correspondences:
            print("Error: no track correspondences found")
            return
        
        # Extract global classification data - it's stored in frame_pair_details
        frame_pair_details = track_results.get('frame_pair_details', {})
        global_inlier_tracks = frame_pair_details.get('global_inlier_tracks', set())
        global_outlier_tracks = frame_pair_details.get('global_outlier_tracks', set())  
        global_track_errors = frame_pair_details.get('global_track_errors', {})
        
        # DEBUG: Print what we're actually getting
        print(f"DEBUG - Visualization received:")
        print(f"  track_results keys: {list(track_results.keys())}")
        print(f"  frame_pair_details keys: {list(frame_pair_details.keys())}")
        print(f"  global_inlier_tracks type: {type(global_inlier_tracks)}, length: {len(global_inlier_tracks)}")
        print(f"  global_outlier_tracks type: {type(global_outlier_tracks)}, length: {len(global_outlier_tracks)}")
        print(f"  global_track_errors length: {len(global_track_errors)}")
        print(f"  Sample global inliers: {list(global_inlier_tracks)[:5] if global_inlier_tracks else 'None'}")
        print(f"  Sample global outliers: {list(global_outlier_tracks)[:5] if global_outlier_tracks else 'None'}")
        
    else:
        print("Error: Expected frame-to-frame results not found")
        print(f"Available keys: {list(all_results.keys())}")
        return
    
    print(f"Found {len(track_correspondences)} total tracks")
    print(f"Track correspondence keys (first 10): {list(track_correspondences.keys())[:10]}")
    print(f"Global inliers: {len(global_inlier_tracks)}, Global outliers: {len(global_outlier_tracks)}")
    print(f"Global inlier sample: {list(global_inlier_tracks)[:5] if global_inlier_tracks else 'None'}")
    print(f"Global outlier sample: {list(global_outlier_tracks)[:5] if global_outlier_tracks else 'None'}")
    print(f"Weighted average reprojection error: {weighted_error:.3f} pixels")
    
    # Use all frames in the video
    all_frame_indices = list(range(len(frames)))
    min_frame = 0
    max_frame = len(frames) - 1
    
    print(f"Creating GIF with {len(all_frame_indices)} frames (from {min_frame} to {max_frame})")
    
    # Store all frames
    gif_frames = []
    successful_frames = 0
    skipped_frames = 0
    
    for i, frame_idx in enumerate(all_frame_indices):
        if i % 10 == 0:  # Print progress every 10 frames
            print(f"Processing frame {frame_idx} ({i+1}/{len(all_frame_indices)})")
        
        try:
            # Check if frame index is valid
            if frame_idx >= len(frames) or frame_idx < 0:
                print(f"Skipping frame {frame_idx}: index out of range (total frames: {len(frames)})")
                skipped_frames += 1
                continue
            
            # Check if frame is None or empty
            if frames[frame_idx] is None or frames[frame_idx].size == 0:
                print(f"Skipping frame {frame_idx}: frame is None or empty")
                skipped_frames += 1
                continue
            
            # Create the plot for this frame using global inlier/outlier classification
            pil_image = create_track_frame_plot_global(
                frames, pred_tracks, pred_visibility, 
                track_correspondences, global_inlier_tracks, global_outlier_tracks,
                global_track_errors, frame_idx, figsize
            )
            
            if pil_image is not None:
                gif_frames.append(pil_image)
                successful_frames += 1
            else:
                print(f"Skipping frame {frame_idx}: failed to create plot")
                skipped_frames += 1
                
        except Exception as e:
            print(f"Skipping frame {frame_idx}: unexpected error - {e}")
            skipped_frames += 1
            continue
    
    # Save as GIF
    if gif_frames:
        try:
            gif_frames[0].save(
                output_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration,
                loop=0
            )
            print(f"GIF saved as: {output_path}")
            print(f"Successfully processed frames: {successful_frames}")
            print(f"Skipped frames: {skipped_frames}")
            print(f"Total frames in GIF: {len(gif_frames)}")
        except Exception as e:
            print(f"Error saving GIF: {e}")
    else:
        print("No frames were successfully processed! GIF not created.")
        print(f"Total skipped frames: {skipped_frames}")

def create_track_frame_plot_global(frames, pred_tracks, pred_visibility, track_correspondences, 
                                  global_inlier_tracks, global_outlier_tracks, global_track_errors,
                                  frame_idx, figsize=(12, 8)):
    """
    Create a plot for a single frame showing globally classified inlier vs outlier tracks
    
    Args:
        frames: Array of video frames
        pred_tracks: Predicted tracks tensor
        pred_visibility: Predicted visibility tensor
        track_correspondences: Dictionary mapping track IDs to correspondences
        global_inlier_tracks: Set of track IDs that are global inliers
        global_outlier_tracks: Set of track IDs that are global outliers
        global_track_errors: Dictionary mapping track IDs to their average errors
        frame_idx: Current frame index
        figsize: Figure size
    
    Returns:
        PIL Image of the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import io
    import torch
    import numpy as np
    
    try:
        # Get frame dimensions
        frame_height, frame_width = frames[frame_idx].shape[:2]
        
        # Create figure with fixed DPI and size
        dpi = 100
        fig_width = frame_width / dpi
        fig_height = (frame_height + 100) / dpi  # Add space for title
        
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)
        
        # Display the frame with exact pixel mapping
        ax.imshow(frames[frame_idx], aspect='equal')
        ax.set_title(f'Frame {frame_idx}: Global Inliers vs Outliers', fontsize=14, pad=10)
        
        # Set exact axis limits to prevent resizing
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)  # Inverted Y-axis for image coordinates
        ax.axis('off')
        
        # Remove all margins and padding
        plt.subplots_adjust(left=0, bottom=0, right=1, top=0.92, wspace=0, hspace=0)
        
        # Convert tensors to numpy if needed
        if isinstance(pred_tracks, torch.Tensor):
            tracks_np = pred_tracks[0].cpu().numpy()  # Remove batch dimension
            visibility_np = pred_visibility[0].cpu().numpy()  # Remove batch dimension
        else:
            tracks_np = pred_tracks[0] if pred_tracks.ndim == 4 else pred_tracks
            visibility_np = pred_visibility[0] if pred_visibility.ndim == 3 else pred_visibility
        
        # Check frame index bounds
        if frame_idx >= tracks_np.shape[0] or frame_idx >= visibility_np.shape[0]:
            print(f"Frame {frame_idx} is out of bounds for tracks data")
            plt.close(fig)
            return None
        
        # Get all tracks that have correspondences and are visible in this frame
        visible_tracks = []
        for track_id in track_correspondences.keys():
            # Check bounds
            if (track_id < tracks_np.shape[1] and 
                track_id < visibility_np.shape[1] and
                visibility_np[frame_idx, track_id] > 0.5):
                visible_tracks.append(track_id)
        
        if len(visible_tracks) == 0:
            print(f"Warning: No visible tracks in frame {frame_idx}")
            print(f"  Track correspondences keys: {list(track_correspondences.keys())[:10]}")
            print(f"  Tracks shape: {tracks_np.shape}")
            print(f"  Visibility shape: {visibility_np.shape}")
        
        # Debug: Print classification info
        visible_inliers = [t for t in visible_tracks if t in global_inlier_tracks]
        visible_outliers = [t for t in visible_tracks if t in global_outlier_tracks]
        
        print(f"Frame {frame_idx} debug:")
        print(f"  Total visible tracks: {len(visible_tracks)}")
        print(f"  Visible inliers: {len(visible_inliers)} - {visible_inliers[:5]}")
        print(f"  Visible outliers: {len(visible_outliers)} - {visible_outliers[:5]}")
        print(f"  Global inlier set size: {len(global_inlier_tracks)}")
        print(f"  Global outlier set size: {len(global_outlier_tracks)}")
        
        # Plot tracks using global classification (same color in all frames)
        inlier_count = 0
        outlier_count = 0
        unused_count = 0
        
        for track_id in visible_tracks:
            try:
                # Get track position at this frame
                track_pos = tracks_np[frame_idx, track_id]
                x, y = float(track_pos[0]), float(track_pos[1])
                
                # Skip if coordinates are invalid
                if not (0 <= x < frames[frame_idx].shape[1] and 0 <= y < frames[frame_idx].shape[0]):
                    continue
                
                # Use global classification (consistent across all frames)
                if track_id in global_inlier_tracks:
                    # Global inlier - green circle
                    circle = patches.Circle((x, y), radius=4, facecolor='lime', 
                                          edgecolor='darkgreen', linewidth=2, alpha=0.8)
                    ax.add_patch(circle)
                    inlier_count += 1
                elif track_id in global_outlier_tracks:
                    # Global outlier - red X
                    ax.plot(x, y, 'rx', markersize=10, markeredgewidth=3, alpha=0.8)
                    outlier_count += 1
                else:
                    # Track visible but not classified (no correspondences)
                    ax.plot(x, y, 'o', color='lightgray', markersize=4, alpha=0.6)
                    unused_count += 1
                    
            except Exception as e:
                print(f"Error plotting track {track_id} in frame {frame_idx}: {e}")
                continue
        
        # Add legend with fixed position
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
                   markeredgecolor='darkgreen', markersize=10, label=f'Global Inliers ({inlier_count})'),
            Line2D([0], [0], marker='x', color='red', markersize=10, 
                   markeredgewidth=3, label=f'Global Outliers ({outlier_count})', linestyle='None'),
            Line2D([0], [0], marker='o', color='lightgray', markersize=6, 
                   label=f'Unused ({unused_count})', linestyle='None')
        ]
        
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                          fancybox=True, shadow=True, framealpha=0.9)
        legend.set_bbox_to_anchor((0.98, 0.98))
        
        # Add text box with global statistics
        total_global_inliers = len(global_inlier_tracks)
        total_global_outliers = len(global_outlier_tracks)
        
        # Show error statistics for visible tracks
        visible_inlier_errors = [global_track_errors.get(tid, 0) for tid in visible_tracks if tid in global_inlier_tracks]
        visible_outlier_errors = [global_track_errors.get(tid, 0) for tid in visible_tracks if tid in global_outlier_tracks]
        
        textstr = (f'Frame: {frame_idx}\n'
                  f'Visible: {len(visible_tracks)}\n'
                  f'Inliers: {inlier_count} (total: {total_global_inliers})\n'
                  f'Outliers: {outlier_count} (total: {total_global_outliers})\n'
                  f'Unused: {unused_count}')
        
        if visible_inlier_errors:
            textstr += f'\nInlier errors: {np.mean(visible_inlier_errors):.2f}±{np.std(visible_inlier_errors):.2f}'
        if visible_outlier_errors:  
            textstr += f'\nOutlier errors: {np.mean(visible_outlier_errors):.2f}±{np.std(visible_outlier_errors):.2f}'
        
        props = dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.9)
        text_box = ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                          verticalalignment='top', bbox=props)
        
        # Convert to PIL Image with consistent settings
        buf = io.BytesIO()
        plt.savefig(buf, format='png', 
                   bbox_inches='tight',
                   dpi=dpi, 
                   facecolor='white', 
                   edgecolor='none',
                   pad_inches=0.1)
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)
        
        return pil_image
        
    except Exception as e:
        print(f"Error creating plot for frame {frame_idx}: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None
    


def save_multi_frame_points_for_web_with_colored_tracks(frames, model, device, output_path, 
                                   frame_indices=None, max_points_per_frame=20000, verbose=True, 
                                   pred_tracks=None, pred_visibility=None, track_labels=None):
    """
    Save 3D points from multiple frames for web visualization with colored track points based on cluster labels
    
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
    """
    import torch
    import numpy as np
    import json
    
    if frame_indices is None:
        frame_indices = list(range(len(frames)))
    
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
    
    # Pre-compute 3D coordinates for all frames to enable arrow creation
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
                'mask': mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
            }
        except Exception as e:
            if verbose:
                print(f"  Error computing 3D coords for frame {frame_idx}: {e}")
            frame_3d_coords[frame_idx] = None
    
    if verbose:
        print(f"Processing {len(frame_indices)} frames with colored track points...")
    
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
            
            # Extract valid points and set WHITE colors for all depth points
            valid_points = points_np[valid_indices]
            depth_colors = np.ones((len(valid_points), 3), dtype=np.float32)  # All white [1.0, 1.0, 1.0]
            
            # Subsample depth points if too many
            if len(valid_points) > max_points_per_frame:
                indices = np.random.choice(len(valid_points), max_points_per_frame, replace=False)
                valid_points = valid_points[indices]
                depth_colors = depth_colors[indices]
            
            # Process track points with cluster colors
            track_arrows = []
            track_points = []
            track_point_colors = []
            
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
                        
                        # Assign colors based on cluster labels
                        for j, track_idx in enumerate(valid_track_indices):
                            if track_idx < len(track_labels):  # Safety check
                                label = track_labels[track_idx]
                                color = cluster_colors.get(label, [0.5, 0.5, 0.5])  # Default gray if label not found
                                
                                track_points.append(current_3d[j].tolist())
                                track_point_colors.append(color)
                
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
            
            if verbose and frame_idx == frame_indices[0]:
                print(f"Frame {frame_idx} final summary:")
                print(f"  Depth points (white): {len(valid_points)}")  
                print(f"  Track points (colored): {len(track_points)}")
                print(f"  Track arrows (red): {len(track_arrows)}")
            
            # Combine all points and colors
            all_points = valid_points.tolist() + track_points
            all_colors = depth_colors.tolist() + track_point_colors
            
            # Update global bounds
            if all_points:
                combined_points = np.array(all_points)
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
                "arrow_count": len(track_arrows)
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
        "frame_indices": [frame["frame_index"] for frame in all_frames_data],
        "has_arrows": tracks_np is not None,
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
        print(f"\nSaved multi-frame point cloud with colored tracks to {output_path}")
        print(f"Total frames: {len(all_frames_data)}")
        print(f"Total depth points (white): {total_depth_points:,}")
        print(f"Total track points (colored): {total_track_points:,}")
        if tracks_np is not None:
            print(f"Total track arrows: {total_arrows:,}")
        if track_labels is not None:
            n_clusters = len(np.unique(track_labels[track_labels != -1]))
            n_error_tracks = np.sum(track_labels == -1)
            print(f"Cluster summary: {n_clusters} valid clusters, {n_error_tracks} error tracks")
        print(f"Global bounds:")
        print(f"  X[{global_bounds['min_x']:.2f}, {global_bounds['max_x']:.2f}]")
        print(f"  Y[{global_bounds['min_y']:.2f}, {global_bounds['max_y']:.2f}]")
        print(f"  Z[{global_bounds['min_z']:.2f}, {global_bounds['max_z']:.2f}]")



# threshold_2nd_iter = best_threshold
# create_inliers_outliers_gif(
#     frames=frames,
#     pred_tracks=pred_tracks, 
#     pred_visibility=pred_visibility,
#     all_results=all_results,
#     output_path=f'MOGE_GLOBAL_RESULTS/BISECTORyoutube_{vid}_results_threshold_{best_threshold}_grid_size_{grid_size}_num_init_{num_initializations}_res_freq_{result_frequency}_2nditerthresh_{threshold_2nd_iter}.gif',
#     duration=600,
#     figsize=(12, 8)
# )