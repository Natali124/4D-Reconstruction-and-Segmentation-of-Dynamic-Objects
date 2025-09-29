import torch
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def create_inliers_outliers_gif(frames, pred_tracks, pred_visibility, all_results, 
                               output_path='inliers_outliers.gif', duration=300, figsize=(12, 8)):
    """
    Create a GIF showing inliers vs outliers across all frames
    
    Args:
        frames: Array of video frames
        pred_tracks: Predicted tracks tensor
        pred_visibility: Predicted visibility tensor  
        all_results: Dictionary with results for each frame pair
        output_path: Output GIF file path
        duration: Duration per frame in milliseconds
        figsize: Figure size for each frame
    """
    
    # Get all frame pairs and determine the range of frames to plot
    frame_pairs = list(all_results.keys())
    if not frame_pairs:
        print("No frame pairs to process!")
        return
    
    # Find the range of all frames to plot
    min_frame = min([pair[0] for pair in frame_pairs])
    max_frame = max([pair[1] for pair in frame_pairs])
    all_frame_indices = list(range(min_frame, max_frame + 1))
    
    print(f"Creating GIF with {len(all_frame_indices)} frames (from {min_frame} to {max_frame})")
    
    # Store all frames
    gif_frames = []
    
    for i, frame_idx in enumerate(all_frame_indices):
        print(f"Processing frame {frame_idx} ({i+1}/{len(all_frame_indices)})")
        
        # Find which frame pair this frame belongs to
        results = None
        for frames_pair, result in all_results.items():
            if frames_pair[0] <= frame_idx <= frames_pair[1]:
                results = result
                break
        
        if results is None:
            print(f"No results found for frame {frame_idx}")
            continue
        
        # Create the plot for this frame using the results from its frame pair
        pil_image = create_frame_plot(
            frames, pred_tracks, pred_visibility, 
            results[2]['valid_indices'], results[2]['inliers'], 
            frame_idx, figsize, frames_pair
        )
        
        if pil_image is not None:
            gif_frames.append(pil_image)
    
    # Save as GIF
    if gif_frames:
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved as: {output_path}")
        print(f"Total frames: {len(gif_frames)}")
    else:
        print("No frames were successfully processed!")

def create_frame_plot(frames, pred_tracks, pred_visibility, valid_indices, inliers, frame_idx, figsize, frames_pair):
    """
    Create a single frame plot showing inliers vs outliers
    Returns PIL Image
    """
    try:
        # Convert inliers to boolean mask if needed
        if isinstance(inliers, np.ndarray) and inliers.dtype == bool:
            inlier_mask = inliers
        else:
            inlier_mask = np.zeros(len(valid_indices), dtype=bool)
            inlier_mask[inliers] = True
        
        # Get the actual track indices
        valid_track_indices = np.array(valid_indices)
        inlier_track_indices = valid_track_indices[inlier_mask]
        outlier_track_indices = valid_track_indices[~inlier_mask]
        
        # Get tracks for this frame
        tracks_frame = pred_tracks[0, frame_idx]  # (N, 2)
        visibility_frame = pred_visibility[0, frame_idx] > 0.5  # (N,) boolean
        
        # Get coordinates for inliers and outliers that are visible
        inlier_coords = []
        outlier_coords = []
        
        for track_idx in inlier_track_indices:
            if visibility_frame[track_idx]:
                x, y = tracks_frame[track_idx].cpu().numpy()
                inlier_coords.append([x, y])
        
        for track_idx in outlier_track_indices:
            if visibility_frame[track_idx]:
                x, y = tracks_frame[track_idx].cpu().numpy()
                outlier_coords.append([x, y])
        
        inlier_coords = np.array(inlier_coords)
        outlier_coords = np.array(outlier_coords)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frames[frame_idx])
        
        if len(inlier_coords) > 0:
            ax.scatter(inlier_coords[:, 0], inlier_coords[:, 1], 
                      color='lime', s=30, label=f'Inliers ({len(inlier_coords)})', 
                      edgecolors='black', linewidth=0.5, alpha=0.8)
        
        if len(outlier_coords) > 0:
            ax.scatter(outlier_coords[:, 0], outlier_coords[:, 1], 
                      color='red', s=30, label=f'Outliers ({len(outlier_coords)})', 
                      marker='x', linewidth=2)
        
        ax.set_title(f"Frame {frame_idx} (using results from frames {frames_pair[0]}-{frames_pair[1]})", 
                    fontsize=14, pad=20)
        ax.legend(loc='upper right')
        ax.axis('off')
        
        # Convert plot to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        pil_image = Image.open(buf)
        
        plt.close(fig)
        
        return pil_image
        
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        return None


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import os
import glob

def load_frames_from_directory(directory_path):
    """
    Load frames from JPG files in the specified directory
    
    Args:
        directory_path: Path to directory containing JPG files
        
    Returns:
        List of loaded images
    """
    # Get all JPG files and sort them
    jpg_files = glob.glob(os.path.join(directory_path, "*.jpg"))
    jpg_files.extend(glob.glob(os.path.join(directory_path, "*.jpeg")))
    jpg_files.extend(glob.glob(os.path.join(directory_path, "*.JPG")))
    jpg_files.extend(glob.glob(os.path.join(directory_path, "*.JPEG")))
    
    if not jpg_files:
        raise ValueError(f"No JPG files found in {directory_path}")
    
    # Sort files numerically if possible, otherwise alphabetically
    try:
        # Try to extract numbers from filenames for proper sorting
        jpg_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    except:
        jpg_files.sort()
    
    print(f"Found {len(jpg_files)} JPG files in {directory_path}")
    
    # Load all images
    frames = []
    for i, jpg_file in enumerate(jpg_files):
        try:
            img = Image.open(jpg_file)
            frames.append(np.array(img))
            if i % 10 == 0:  # Progress update every 10 files
                print(f"Loaded {i+1}/{len(jpg_files)} images...")
        except Exception as e:
            print(f"Error loading {jpg_file}: {e}")
    
    print(f"Successfully loaded {len(frames)} frames")
    return frames

def create_debug_gif(frames, start_idx, end_idx, output_path='debug_frames.gif', 
                    duration=500, figsize=(10, 8), add_frame_numbers=True):
    # Load frames if path is provided
    if isinstance(frames, str):
        frames = load_frames_from_directory(frames)
    
    # Validate indices
    if start_idx < 0 or end_idx >= len(frames) or start_idx > end_idx:
        raise ValueError(f"Invalid indices: start_idx={start_idx}, end_idx={end_idx}, total_frames={len(frames)}")
    
    print(f"Creating debug GIF for frames {start_idx} to {end_idx}")
    print(f"Total frames to process: {end_idx - start_idx + 1}")
    
    gif_frames = []
    
    for frame_idx in range(start_idx, end_idx + 1):
        print(f"Processing frame {frame_idx}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frames[frame_idx])
        
        if add_frame_numbers:
            # Add frame number as text overlay
            ax.text(10, 30, f"Frame {frame_idx}", 
                   fontsize=16, color='white', weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        ax.axis('off')
        ax.set_title(f"Frame {frame_idx}", fontsize=14, pad=20)
        
        # Convert plot to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        pil_image = Image.open(buf)
        gif_frames.append(pil_image)
        
        plt.close(fig)
    
    # Save as GIF
    if gif_frames:
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"Debug GIF saved as: {output_path}")
        print(f"Frames: {start_idx} to {end_idx} ({len(gif_frames)} total)")
    else:
        print("No frames were processed!")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

def create_inliers_outliers_gif(frames, pred_tracks, pred_visibility, all_results, 
                               output_path='inliers_outliers.gif', duration=300, figsize=(12, 8)):
    """
    Create a GIF showing inliers vs outliers across all frames
    
    Args:
        frames: Array of video frames
        pred_tracks: Predicted tracks tensor
        pred_visibility: Predicted visibility tensor  
        all_results: Dictionary with results for each frame pair
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
    
    # Get all frame pairs and determine the range of frames to plot
    frame_pairs = list(all_results.keys())
    if not frame_pairs:
        print("No frame pairs to process!")
        return
    
    # Find the range of all frames to plot
    try:
        min_frame = min([pair[0] for pair in frame_pairs])
        max_frame = max([pair[1] for pair in frame_pairs])
        all_frame_indices = list(range(min_frame, max_frame + 1))
    except Exception as e:
        print(f"Error determining frame range: {e}")
        return
    
    print(f"Creating GIF with {len(all_frame_indices)} frames (from {min_frame} to {max_frame})")
    
    # Store all frames
    gif_frames = []
    successful_frames = 0
    skipped_frames = 0
    
    for i, frame_idx in enumerate(all_frame_indices):
        print(f"Processing frame {frame_idx} ({i+1}/{len(all_frame_indices)})")
        
        try:
            # Check if frame index is valid
            if frame_idx >= len(frames) or frame_idx < 0:
                print(f"Skipping frame {frame_idx}: index out of range (total frames: {len(frames)})")
                skipped_frames += 1
                continue
            
            # Check if frame is None
            if frames[frame_idx] is None:
                print(f"Skipping frame {frame_idx}: frame is None")
                skipped_frames += 1
                continue
            
            # Find which frame pair this frame belongs to
            results = None
            frames_pair = None
            for pair, result in all_results.items():
                if pair[0] <= frame_idx <= pair[1]:
                    results = result
                    frames_pair = pair
                    break
            
            if results is None:
                print(f"Skipping frame {frame_idx}: no results found")
                skipped_frames += 1
                continue
            
            # Check if results have the expected structure
            if not isinstance(results, (list, tuple)) or len(results) < 3:
                print(f"Skipping frame {frame_idx}: invalid results structure")
                skipped_frames += 1
                continue
            
            if results[2] is None or 'valid_indices' not in results[2] or 'inliers' not in results[2]:
                print(f"Skipping frame {frame_idx}: missing valid_indices or inliers in results")
                skipped_frames += 1
                continue
            
            # Create the plot for this frame using the results from its frame pair
            pil_image = create_frame_plot(
                frames, pred_tracks, pred_visibility, 
                results[2]['valid_indices'], results[2]['inliers'], 
                frame_idx, figsize, frames_pair
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

def create_frame_plot(frames, pred_tracks, pred_visibility, valid_indices, inliers, frame_idx, figsize, frames_pair):
    """
    Create a single frame plot showing inliers vs outliers
    Returns PIL Image or None if failed
    """
    try:
        # Check for None inputs
        if frames is None or pred_tracks is None or pred_visibility is None:
            print(f"Frame {frame_idx}: One or more inputs is None")
            return None
        
        if valid_indices is None or inliers is None:
            print(f"Frame {frame_idx}: valid_indices or inliers is None")
            return None
        
        # Check frame index bounds
        if frame_idx >= len(frames) or frame_idx < 0:
            print(f"Frame {frame_idx}: index out of bounds")
            return None
        
        # Check if frame is None
        if frames[frame_idx] is None:
            print(f"Frame {frame_idx}: frame data is None")
            return None
        
        # Convert inliers to boolean mask if needed
        if isinstance(inliers, np.ndarray) and inliers.dtype == bool:
            inlier_mask = inliers
        else:
            if len(valid_indices) == 0:
                print(f"Frame {frame_idx}: no valid indices")
                return None
            inlier_mask = np.zeros(len(valid_indices), dtype=bool)
            if len(inliers) > 0:  # Check if inliers is not empty
                inlier_mask[inliers] = True
        
        # Get the actual track indices
        valid_track_indices = np.array(valid_indices)
        inlier_track_indices = valid_track_indices[inlier_mask]
        outlier_track_indices = valid_track_indices[~inlier_mask]
        
        # Check pred_tracks dimensions
        if pred_tracks.shape[1] <= frame_idx:
            print(f"Frame {frame_idx}: frame_idx exceeds pred_tracks dimensions")
            return None
        
        # Get tracks for this frame
        tracks_frame = pred_tracks[0, frame_idx]  # (N, 2)
        visibility_frame = pred_visibility[0, frame_idx] > 0.5  # (N,) boolean
        
        # Get coordinates for inliers and outliers that are visible
        inlier_coords = []
        outlier_coords = []
        
        for track_idx in inlier_track_indices:
            if track_idx < len(visibility_frame) and visibility_frame[track_idx]:
                try:
                    x, y = tracks_frame[track_idx].cpu().numpy()
                    if not (np.isnan(x) or np.isnan(y)):  # Check for NaN values
                        inlier_coords.append([x, y])
                except Exception as e:
                    print(f"Frame {frame_idx}: Error getting inlier coords for track {track_idx}: {e}")
                    continue
        
        for track_idx in outlier_track_indices:
            if track_idx < len(visibility_frame) and visibility_frame[track_idx]:
                try:
                    x, y = tracks_frame[track_idx].cpu().numpy()
                    if not (np.isnan(x) or np.isnan(y)):  # Check for NaN values
                        outlier_coords.append([x, y])
                except Exception as e:
                    print(f"Frame {frame_idx}: Error getting outlier coords for track {track_idx}: {e}")
                    continue
        
        inlier_coords = np.array(inlier_coords) if inlier_coords else np.empty((0, 2))
        outlier_coords = np.array(outlier_coords) if outlier_coords else np.empty((0, 2))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frames[frame_idx])
        
        if len(inlier_coords) > 0:
            ax.scatter(inlier_coords[:, 0], inlier_coords[:, 1], 
                      color='lime', s=30, label=f'Inliers ({len(inlier_coords)})', 
                      edgecolors='black', linewidth=0.5, alpha=0.8)
        
        if len(outlier_coords) > 0:
            ax.scatter(outlier_coords[:, 0], outlier_coords[:, 1], 
                      color='red', s=30, label=f'Outliers ({len(outlier_coords)})', 
                      marker='x', linewidth=2)
        
        title = f"Frame {frame_idx}"
        if frames_pair is not None:
            title += f" (using results from frames {frames_pair[0]}-{frames_pair[1]})"
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(loc='upper right')
        ax.axis('off')
        
        # Convert plot to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        pil_image = Image.open(buf)
        
        plt.close(fig)
        
        return pil_image
        
    except Exception as e:
        print(f"Error processing frame {frame_idx}: {e}")
        try:
            plt.close('all')  # Clean up any open figures
        except:
            pass
        return None
    

def create_modified_visibility(pred_visibility, all_results):
    modified_pred_visibility = pred_visibility.clone()
    inlier_track_indices = set()
    
    for (frame1_idx, frame2_idx), (R, t, results) in all_results.items():
        if R is not None and results is not None:
            # Get the valid indices (tracks that had valid 3D-2D correspondences)
            valid_indices = results['valid_indices']
            
            # Get the inliers (subset of valid_indices that were deemed inliers by RANSAC)
            inliers = results['inliers']
            
            if len(inliers) > 0:
                # Map inliers back to original track indices
                original_track_indices = [valid_indices[i] for i in inliers]
                inlier_track_indices.update(original_track_indices)
                
                # Set visibility to False for these tracks in both frames
                for track_idx in original_track_indices:
                    for frame_idx in range(frame1_idx, frame2_idx+1):
                        modified_pred_visibility[0, frame1_idx, track_idx] = False
                        modified_pred_visibility[0, frame2_idx, track_idx] = False
    
    
    return modified_pred_visibility
