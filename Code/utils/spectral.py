from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
import numpy as np

def get_3d_coordinates_for_frame(frames, model, device, frame_idx):
    """Get 3D coordinates and intrinsics for a specific frame using MoGe"""
    input_image = frames[frame_idx]
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    output = model.infer(input_image)
    return output["points"], output["mask"], output["intrinsics"]

def spectral_cluster_trajectories(
   trajs: np.ndarray,
   normalize_mode: str = "center",   
   k_min: int = 2,
   k_max: int | None = None,
   local_scale_k: int | None = 7,    
   random_state: int = 0,
   return_details: bool = False,
):
   """
   Spectral clustering for 3D trajectories shaped (T, N, 3), with automatic K selection.
   Trajectories containing NaNs are assigned class -1.
   """
   if trajs.ndim != 3 or trajs.shape[-1] != 3:
       raise ValueError("trajs must have shape (frames, n_traj, 3)")
   T, N, D = trajs.shape
   if N == 0:
       return np.array([], dtype=int), 0, {} if return_details else (np.array([], dtype=int), 0)
   if N == 1:
       return np.array([0], dtype=int), 1, {} if return_details else (np.array([0], dtype=int), 1)

   # Check for NaN trajectories
   nan_mask = np.isnan(trajs).any(axis=(0, 2))  # True if trajectory contains any NaN
   valid_mask = ~nan_mask
   n_valid = np.sum(valid_mask)
   
   # If no valid trajectories, return all -1
   if n_valid == 0:
       labels = np.full(N, -1, dtype=int)
       return (labels, 0, {} if return_details else (labels, 0))
   
   # If only one valid trajectory, assign it class 0 and rest -1
   if n_valid == 1:
       labels = np.full(N, -1, dtype=int)
       labels[valid_mask] = 0
       return (labels, 1, {} if return_details else (labels, 1))

   X = trajs[:, valid_mask, :].astype(float).copy()  # (T, n_valid, 3)
   
   if normalize_mode == "center":
       X -= X.mean(axis=0, keepdims=True)
   elif normalize_mode == "start":
       X -= X[0:1, :, :]
   elif normalize_mode == "none":
       pass
   else:
       raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

   V = X.transpose(1, 0, 2).reshape(n_valid, T * D)  # (n_valid, 3T)

   # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
   G = V @ V.T
   sq_norms = np.sum(V * V, axis=1, keepdims=True)  # (n_valid,1)
   D2 = sq_norms + sq_norms.T - 2 * G
   np.maximum(D2, 0, out=D2) 
   Dmat = np.sqrt(D2) / np.sqrt(T)  # (n_valid, n_valid)

   if local_scale_k is not None and n_valid > 1:
       k = int(np.clip(local_scale_k, 1, max(1, n_valid - 1)))
       # distance to k-th nearest neighbor (exclude self at 0)
       D_no_self = Dmat.copy()
       np.fill_diagonal(D_no_self, np.inf)
       sortD = np.sort(D_no_self, axis=1)
       sigmas = sortD[:, k-1]

       sigmas[sigmas == 0] = np.median(sortD[sortD < np.inf]) if np.any(sortD < np.inf) else 1.0
       denom = (sigmas[:, None] * sigmas[None, :]) + 1e-12
       A = np.exp(-(Dmat * Dmat) / denom)
   else:
       tri = Dmat[np.triu_indices(n_valid, k=1)]
       med = np.median(tri[tri > 0]) if np.any(tri > 0) else (np.median(tri) if tri.size else 1.0)
       sigma = med if med > 0 else 1.0
       A = np.exp(-(Dmat * Dmat) / (2.0 * sigma * sigma))

   np.fill_diagonal(A, 0.0)

   A = 0.5 * (A + A.T)
   d = A.sum(axis=1)

   d[d == 0] = 1e-12
   D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
   L_sym = np.eye(n_valid) - D_inv_sqrt @ A @ D_inv_sqrt

   evals, evecs = np.linalg.eigh(L_sym)
   # sort (should already be ascending)
   order = np.argsort(evals)
   evals = evals[order]
   evecs = evecs[:, order]

   if k_max is None:
       k_max = min(10, n_valid)
   k_max = max(k_min, min(k_max, n_valid))  
   if n_valid <= k_min:
       K = n_valid
   else:
       upper = min(k_max, len(evals) - 1)
       gaps = evals[1:upper + 1] - evals[:upper]
       
       start_idx = max(k_min - 1, 0)
       if start_idx >= len(gaps):
           K = min(k_max, n_valid)
       else:
           masked = gaps.copy()
           masked[:start_idx] = -np.inf
           idx = int(np.argmax(masked))
           K = idx + 1
           K = int(np.clip(K, k_min, k_max))

   if K <= 1:
       valid_labels = np.zeros(n_valid, dtype=int)
   else:
       U = evecs[:, :K]
       U = normalize(U, norm="l2", axis=1)

       km = KMeans(n_clusters=K, n_init=20, random_state=random_state)
       valid_labels = km.fit_predict(U)

   # Create final labels array
   labels = np.full(N, -1, dtype=int)
   labels[valid_mask] = valid_labels

   if return_details:
       details = {
           "eigvals": evals,
           "eigvecs": evecs,
           "embedding": U if K > 1 else np.zeros((n_valid, 1)),
           "affinity": A,
           "distances": Dmat,
           "valid_mask": valid_mask,
       }
       return labels, K, details
   else:
       return labels, K
   
def get_trajectories(frames, model, device, pred_tracks, pred_visibility, debug=False):
    """Get 3D trajectories for all tracks across all frames with linear interpolation"""
    n_frames = len(frames)
    tracks_np = pred_tracks[0].cpu().numpy()  # Shape: (T, N, 2)
    visibility_np = pred_visibility[0].cpu().numpy()  # Shape: (T, N)
    n_tracks = tracks_np.shape[1]
    
    trajectories = np.full((n_frames, n_tracks, 3), np.nan)
    
    if debug:
        print(f"Total tracks: {n_tracks}")
        print(f"Total frames: {n_frames}")
    
    # First pass: fill in valid points
    for frame_idx in range(n_frames):
        points_3d, mask, _ = get_3d_coordinates_for_frame(frames, model, device, frame_idx)
        points_3d_np = points_3d.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        frame_valid_count = 0
        for track_id in range(n_tracks):
            if visibility_np[frame_idx, track_id] > 0.5:
                x, y = tracks_np[frame_idx, track_id]
                col = int(np.clip(x, 0, points_3d_np.shape[1] - 1))
                row = int(np.clip(y, 0, points_3d_np.shape[0] - 1))
                
                if mask_np[row, col]:
                    trajectories[frame_idx, track_id] = points_3d_np[row, col]
                    frame_valid_count += 1
        
        if debug and frame_idx < 3:
            print(f"Frame {frame_idx}: {frame_valid_count} valid 3D points")
    
    # Count tracks with at least one valid point before interpolation
    tracks_with_valid_points = 0
    for track_id in range(n_tracks):
        traj = trajectories[:, track_id, :]
        valid_mask = ~np.isnan(traj).any(axis=1)
        if valid_mask.any():
            tracks_with_valid_points += 1
    
    if debug:
        print(f"Tracks with at least one valid 3D point: {tracks_with_valid_points}/{n_tracks}")
    
    # Second pass: interpolate missing points for each track
    for track_id in range(n_tracks):
        traj = trajectories[:, track_id, :]
        valid_mask = ~np.isnan(traj).any(axis=1)
        
        if not valid_mask.any():
            continue
            
        valid_indices = np.where(valid_mask)[0]
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]
        
        # Fill missing points before first valid point
        if first_valid > 0:
            trajectories[:first_valid, track_id] = trajectories[first_valid, track_id]
        
        # Fill missing points after last valid point
        if last_valid < n_frames - 1:
            trajectories[last_valid+1:, track_id] = trajectories[last_valid, track_id]
        
        # Interpolate gaps between valid points
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            
            if end_idx - start_idx > 1:  # There's a gap
                start_point = trajectories[start_idx, track_id]
                end_point = trajectories[end_idx, track_id]
                
                for gap_idx in range(start_idx + 1, end_idx):
                    alpha = (gap_idx - start_idx) / (end_idx - start_idx)
                    trajectories[gap_idx, track_id] = (1 - alpha) * start_point + alpha * end_point
    
    return trajectories

def run_spectral(frames, vid, model, device, output_path):
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
    grid_size = 20
    num_initializations = 5
    verbose = False

    pred_tracks, pred_visibility = get_tracks_and_visibility(video, grid_size=grid_size, num_initializations = num_initializations, device=device, verbose=verbose)

    traj = get_trajectories(frames, model, device, pred_tracks, pred_visibility, debug=True)

    labels, K, details = spectral_cluster_trajectories(traj, normalize_mode="center",return_details=True)

    create_spectral_clustering_gif_fast(
        frames=frames,
        pred_tracks=pred_tracks,
        pred_visibility=pred_visibility,
        labels=labels,
        output_path=output_path + '/spectral.gif',
        duration=400,  # Slightly slower for better viewing
    )

    return labels

from pnp_RANSAC_first_frame import get_tracks_and_visibility

def create_spectral_clustering_gif_fast(frames, pred_tracks, pred_visibility, labels, 
                                       output_path='spectral_clustering.gif', duration=300, 
                                       max_frames=None, skip_frames=1):
    """
    Fast version of spectral clustering GIF creation - HUGE TITLE and BIG POINTS, NO LEGEND
    
    Args:
        frames: Array of video frames
        pred_tracks: Predicted tracks tensor
        pred_visibility: Predicted visibility tensor
        labels: Cluster labels from spectral clustering
        output_path: Output GIF file path
        duration: Duration per frame in milliseconds
        max_frames: Maximum number of frames to process (None = all)
        skip_frames: Skip every N frames (1 = use all frames)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import io
    
    if frames is None or pred_tracks is None or pred_visibility is None or labels is None:
        print("Error: Missing required inputs")
        return
    
    # Convert tensors once at the beginning
    if hasattr(pred_tracks, 'cpu'):
        tracks_np = pred_tracks[0].cpu().numpy()
        visibility_np = pred_visibility[0].cpu().numpy()
    else:
        tracks_np = pred_tracks[0] if pred_tracks.ndim == 4 else pred_tracks
        visibility_np = pred_visibility[0] if pred_visibility.ndim == 3 else pred_visibility
    
    # Prepare frame indices
    frame_indices = list(range(0, len(frames), skip_frames))
    if max_frames:
        frame_indices = frame_indices[:max_frames]
    
    print(f"Processing {len(frame_indices)} frames (skip={skip_frames})")
    
    # Pre-compute colors
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels), 3)))
    label_colors = {label: colors[i] if label != -1 else 'gray' 
                   for i, label in enumerate(unique_labels)}
    
    # Fixed figure setup
    plt.ioff()  # Turn off interactive mode
    frame_height, frame_width = frames[0].shape[:2]
    
    # Calculate figure size to maintain aspect ratio
    target_width = max(10, frame_width / 80)
    target_height = target_width * (frame_height / frame_width)
    
    # HUGE title and BIG points settings
    fontsize_title = 48  # HUGE TITLE
    marker_size = 120    # BIG POINTS
    
    gif_frames = []
    
    for i, frame_idx in enumerate(frame_indices):
        if i % 10 == 0:
            print(f"Frame {i+1}/{len(frame_indices)}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(target_width, target_height), dpi=100)
        ax.imshow(frames[frame_idx], origin='upper', aspect='equal')
        ax.set_title(f'Frame {frame_idx}', fontsize=fontsize_title, pad=20)
        ax.axis('off')
        
        # Set consistent axis limits
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)  # Invert Y-axis to match image coordinates
        
        plt.tight_layout(pad=0.5)
        
        # Vectorized visibility check
        visible_mask = visibility_np[frame_idx, :len(labels)] > 0.5
        visible_tracks = np.where(visible_mask)[0]
        
        if len(visible_tracks) > 0:
            # Get positions for all visible tracks at once
            positions = tracks_np[frame_idx, visible_tracks]
            track_labels = labels[visible_tracks]
            
            # Filter valid positions
            valid_x = (positions[:, 0] >= 0) & (positions[:, 0] < frame_width) & np.isfinite(positions[:, 0])
            valid_y = (positions[:, 1] >= 0) & (positions[:, 1] < frame_height) & np.isfinite(positions[:, 1])
            valid_mask = valid_x & valid_y
            
            if np.any(valid_mask):
                valid_positions = positions[valid_mask]
                valid_labels = track_labels[valid_mask]
                
                # Plot by cluster with BIG points
                for label in unique_labels:
                    mask = valid_labels == label
                    if np.any(mask):
                        color = label_colors[label]
                        pos = valid_positions[mask]
                        
                        if label == -1:  # Noise - BIG x markers
                            ax.scatter(pos[:, 0], pos[:, 1], c=color, marker='x', 
                                     s=marker_size, alpha=0.9, linewidths=3)
                        else:  # Clusters - BIG circles with black edges
                            ax.scatter(pos[:, 0], pos[:, 1], c=color, s=marker_size, alpha=0.9, 
                                     edgecolors='black', linewidths=2)
        
        # Convert to PIL with higher DPI for crisp output
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   pad_inches=0.1, facecolor='white', orientation='portrait')
        buf.seek(0)
        gif_frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)
        plt.close('all')  # Ensure all figures are closed
    
    # Save GIF
    if gif_frames:
        gif_frames[0].save(output_path, save_all=True, append_images=gif_frames[1:], 
                          duration=duration, loop=0, optimize=True, quality=95)
        print(f"Fast GIF saved: {output_path}")

# Ultra-fast version for quick previews
def create_spectral_clustering_gif_ultra_fast(frames, pred_tracks, pred_visibility, labels, 
                                            output_path='spectral_clustering_fast.gif'):
    """Ultra-fast version - minimal quality, maximum speed"""
    
    # Use every 3rd frame, lower resolution, simple plotting
    create_spectral_clustering_gif_fast(
        frames, pred_tracks, pred_visibility, labels,
        output_path=output_path,
        duration=200,
        max_frames=20,  # Only first 20 frames 
        skip_frames=2   # Every other frame
    )

def create_spectral_clustering_frame_plot(frames, pred_tracks, pred_visibility, labels, 
                                        label_to_color, frame_idx, figsize=(12, 8)):
    """
    Create a plot for a single frame showing spectral clustering results - HUGE TITLE and BIG POINTS
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
        
        # Calculate figure size to maintain aspect ratio
        target_width = max(10, frame_width / 80)
        target_height = target_width * (frame_height / frame_width)
        
        # HUGE title and BIG points settings
        fontsize_title = 48  # HUGE TITLE
        marker_size = 120    # BIG POINTS
        
        fig, ax = plt.subplots(1, 1, figsize=(target_width, target_height), dpi=100)
        
        # Display frame
        ax.imshow(frames[frame_idx], origin='upper', aspect='equal')
        ax.set_title(f'Frame {frame_idx}', fontsize=fontsize_title, pad=20)
        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)  # Invert Y-axis to match image coordinates
        ax.axis('off')
        
        plt.tight_layout(pad=0.5)
        
        # Convert tensors to numpy
        if isinstance(pred_tracks, torch.Tensor):
            tracks_np = pred_tracks[0].cpu().numpy()
            visibility_np = pred_visibility[0].cpu().numpy()
        else:
            tracks_np = pred_tracks[0] if pred_tracks.ndim == 4 else pred_tracks
            visibility_np = pred_visibility[0] if pred_visibility.ndim == 3 else pred_visibility
        
        # Check bounds
        if frame_idx >= tracks_np.shape[0] or frame_idx >= visibility_np.shape[0]:
            print(f"Frame {frame_idx} out of bounds")
            plt.close(fig)
            return None
        
        # Plot tracks with cluster colors - BIG points
        for track_id in range(min(tracks_np.shape[1], len(labels))):
            # Check visibility
            if (track_id < visibility_np.shape[1] and 
                visibility_np[frame_idx, track_id] > 0.5):
                
                # Get track position
                track_pos = tracks_np[frame_idx, track_id]
                x, y = float(track_pos[0]), float(track_pos[1])
                
                # Skip invalid coordinates
                if not (0 <= x < frame_width and 0 <= y < frame_height):
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                
                # Get cluster label and color
                cluster_label = labels[track_id]
                color = label_to_color.get(cluster_label, 'black')
                
                # Plot BIG points
                if cluster_label == -1:  # Noise points - BIG X
                    ax.scatter([x], [y], c=color, marker='x', s=marker_size, 
                             alpha=0.9, linewidths=3)
                else:  # Clustered points - BIG circles with black edges
                    ax.scatter([x], [y], c=color, s=marker_size, alpha=0.9,
                             edgecolors='black', linewidths=2)
        
        # Convert to PIL Image with higher DPI
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                   facecolor='white', edgecolor='none', pad_inches=0.1,
                   orientation='portrait')
        buf.seek(0)
        pil_image = Image.open(buf).copy()
        buf.close()
        plt.close(fig)
        plt.close('all')  # Ensure all figures are closed
        
        return pil_image
        
    except Exception as e:
        print(f"Error creating plot for frame {frame_idx}: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None