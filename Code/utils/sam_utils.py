import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from collections import defaultdict
import os
import io

def load_sam_model(model_type="vit_b", checkpoint_path="sam_vit_b_01ec64.pth"):
    """
    Load SAM model using the vit_b checkpoint.
    
    Args:
        model_type: "vit_h", "vit_l", or "vit_b" (default: vit_b)
        checkpoint_path: Path to the SAM checkpoint file (default: sam_vit_b_01ec64.pth)
    
    Returns:
        SAM mask generator
    """
    if not os.path.exists(checkpoint_path):
        print(f"SAM checkpoint not found at: {checkpoint_path}")
        print("Please ensure sam_vit_b_01ec64.pth is in the current directory")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM model (vit_b) from: {checkpoint_path}")
    print(f"Using device: {device}")
    
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    
    return mask_generator

def get_tracks_on_segments(tracks, visibility, segments, frame_idx=0):
    """
    For each segment, find which tracks fall on it.
    
    Args:
        tracks: Predicted tracks [batch, frames, num_tracks, 2] or [frames, num_tracks, 2]
        visibility: Visibility predictions [batch, frames, num_tracks] or [frames, num_tracks]
        segments: List of segment masks from SAM
        frame_idx: Frame index to analyze
    
    Returns:
        Dictionary mapping segment_id to list of track_ids on that segment
    """
    # Handle batch dimension
    if len(tracks.shape) == 4:
        tracks_np = tracks[0].cpu().numpy() if hasattr(tracks, 'cpu') else tracks[0]
        visibility_np = visibility[0].cpu().numpy() if hasattr(visibility, 'cpu') else visibility[0]
    else:
        tracks_np = tracks.cpu().numpy() if hasattr(tracks, 'cpu') else tracks
        visibility_np = visibility.cpu().numpy() if hasattr(visibility, 'cpu') else visibility
    
    tracks_frame = tracks_np[frame_idx]
    visibility_frame = visibility_np[frame_idx]
    
    segment_tracks = {}
    
    for seg_id, segment in enumerate(segments):
        segment_tracks[seg_id] = []
        mask = segment['segmentation']
        H, W = mask.shape
        
        for track_id in range(len(tracks_frame)):
            if visibility_frame[track_id] <= 0.5:
                continue
            
            track_pos = tracks_frame[track_id]
            x = int(np.clip(track_pos[0], 0, W - 1))
            y = int(np.clip(track_pos[1], 0, H - 1))
            
            if mask[y, x]:  # Track falls on this segment
                segment_tracks[seg_id].append(track_id)
    
    return segment_tracks

def filter_segments_by_outliers(segments, segment_tracks, outlier_set, min_outliers=10, outlier_ratio=0.5):
    """
    Filter segments based on outlier criteria.
    
    Args:
        segments: List of segment masks from SAM
        segment_tracks: Dictionary mapping segment_id to list of track_ids
        outlier_set: Set of outlier track IDs
        min_outliers: Minimum number of outliers required
        outlier_ratio: Minimum ratio of outliers to total tracks
    
    Returns:
        List of segment indices that pass the criteria
    """
    kept_segments = []
    
    for seg_id, track_ids in segment_tracks.items():
        if len(track_ids) == 0:
            continue
        
        outliers_on_segment = [tid for tid in track_ids if tid in outlier_set]
        num_outliers = len(outliers_on_segment)
        total_tracks = len(track_ids)
        
        # Check criteria
        has_min_outliers = num_outliers >= min_outliers
        has_outlier_majority = num_outliers / total_tracks >= outlier_ratio
        
        # and or or logical connector
        if has_min_outliers or has_outlier_majority:
            kept_segments.append(seg_id)
            print(f"Segment {seg_id}: {num_outliers}/{total_tracks} outliers ({num_outliers/total_tracks:.2f}) - KEPT")
        else:
            print(f"Segment {seg_id}: {num_outliers}/{total_tracks} outliers ({num_outliers/total_tracks:.2f}) - DISCARDED")
    
    return kept_segments

def visualize_segments_on_frame(frame, segments, kept_segment_ids):
    """
    Simple visualization: just show the frame with kept segment masks overlaid.
    
    Args:
        frame: Image frame [H, W, 3]
        segments: List of all segment masks from SAM
        kept_segment_ids: List of segment IDs to highlight
    
    Returns:
        Visualization image
    """
    # Convert frame to RGB if needed
    if frame.shape[-1] == 3:
        vis_frame = frame.copy()
    else:
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create overlay for segments
    overlay = np.zeros_like(vis_frame)
    colors = plt.cm.Set3(np.linspace(0, 1, len(kept_segment_ids)))
    
    for i, seg_id in enumerate(kept_segment_ids):
        mask = segments[seg_id]['segmentation']
        color = (colors[i][:3] * 255).astype(np.uint8)
        overlay[mask] = color
    
    # Blend frame with overlay (increased opacity)
    alpha = 0.7
    vis_frame = cv2.addWeighted(vis_frame, 1-alpha, overlay, alpha, 0)
    
    return vis_frame

def create_frame_with_title(frame, frame_idx, num_segments):
    """
    Create a matplotlib figure with the frame and a big title.
    
    Args:
        frame: Image frame [H, W, 3]
        frame_idx: Current frame index
        num_segments: Number of segments kept
    
    Returns:
        PIL Image with title
    """
    # Turn off interactive plotting
    plt.ioff()
    
    # Get frame dimensions for figure sizing
    frame_height, frame_width = frame.shape[:2]
    
    # Calculate figure size to maintain aspect ratio
    target_width = max(10, frame_width / 80)
    target_height = target_width * (frame_height / frame_width)
    
    # Create figure with big title
    fig, ax = plt.subplots(figsize=(target_width, target_height), dpi=100)
    
    # Display frame
    ax.imshow(frame, origin='upper', aspect='equal')
    ax.set_title(f'Frame {frame_idx}: {num_segments} Segments Kept', fontsize=48, pad=20)
    ax.axis('off')
    
    # Set consistent axis limits
    ax.set_xlim(0, frame_width)
    ax.set_ylim(frame_height, 0)  # Invert Y-axis to match image coordinates
    
    plt.tight_layout(pad=0.5)
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
               pad_inches=0.1, facecolor='white', orientation='portrait')
    buf.seek(0)
    pil_image = Image.open(buf).copy()
    buf.close()
    
    # Clean up
    plt.close(fig)
    plt.close('all')
    
    return pil_image

def process_frame_with_sam(frame, mask_generator, tracks, visibility, outlier_set, frame_idx, 
                          min_outliers=10, outlier_ratio=0.5):
    """
    Process a single frame with SAM segmentation and outlier filtering.
    
    Args:
        frame: Image frame [H, W, 3]
        mask_generator: SAM mask generator
        tracks: Track predictions
        visibility: Visibility predictions
        outlier_set: Set of outlier track IDs
        frame_idx: Current frame index
        min_outliers: Minimum outliers required
        outlier_ratio: Minimum outlier ratio required
    
    Returns:
        Tuple of (visualization_image, kept_segments_info)
    """
    print(f"\nProcessing frame {frame_idx}...")
    
    # Generate segments with SAM
    segments = mask_generator.generate(frame)
    print(f"SAM generated {len(segments)} segments")
    
    # Find tracks on each segment
    segment_tracks = get_tracks_on_segments(tracks, visibility, segments, frame_idx)
    
    # Filter segments by outlier criteria
    kept_segment_ids = filter_segments_by_outliers(
        segments, segment_tracks, outlier_set, min_outliers, outlier_ratio
    )
    
    print(f"Kept {len(kept_segment_ids)} segments out of {len(segments)}")
    
    # Create simple visualization - just image with masks
    vis_frame = visualize_segments_on_frame(frame, segments, kept_segment_ids)
    
    # Prepare segment info for return
    kept_segments_info = []
    for seg_id in kept_segment_ids:
        track_ids = segment_tracks[seg_id]
        outliers_on_segment = [tid for tid in track_ids if tid in outlier_set]
        kept_segments_info.append({
            'segment_id': seg_id,
            'mask': segments[seg_id]['segmentation'],
            'total_tracks': len(track_ids),
            'outlier_tracks': len(outliers_on_segment),
            'track_ids': track_ids
        })
    
    return vis_frame, kept_segments_info

def run_sam_outlier_analysis(frames, tracks, visibility, outlier_tracks_level0, 
                            sam_checkpoint_path="sam_vit_b_01ec64.pth",
                            min_outliers=10, outlier_ratio=0.5, output_dir="sam_outlier_results"):
    """
    Run complete SAM-based outlier analysis.
    
    Args:
        frames: Video frames [num_frames, H, W, 3] or list of frames
        tracks: Track predictions
        visibility: Visibility predictions  
        outlier_tracks_level0: Set of outlier track IDs (e.g., outlier_tracks[0])
        sam_checkpoint_path: Path to SAM checkpoint file (default: sam_vit_b_01ec64.pth)
        min_outliers: Minimum outliers required per segment
        outlier_ratio: Minimum outlier ratio required per segment
        output_dir: Directory to save results
    
    Returns:
        Dictionary containing results for all frames
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SAM model
    print("Loading SAM model...")
    mask_generator = load_sam_model(checkpoint_path=sam_checkpoint_path)
    if mask_generator is None:
        return None
    
    # Convert outlier tracks to set for faster lookup
    outlier_set = set(outlier_tracks_level0)
    print(f"Analyzing {len(outlier_set)} outlier tracks")
    
    # Process first frame and visualize
    print("\n" + "="*50)
    print("ANALYZING FIRST FRAME")
    print("="*50)
    
    first_frame = frames[0] if isinstance(frames, list) else frames[0]
    first_vis, first_info = process_frame_with_sam(
        first_frame, mask_generator, tracks, visibility, outlier_set, 0, 
        min_outliers, outlier_ratio
    )
    
    # Save first frame visualization with big title
    first_frame_with_title = create_frame_with_title(first_vis, 0, len(first_info))
    first_frame_with_title.save(os.path.join(output_dir, "first_frame_segments.png"))
    
    # Process all frames for GIF
    print(f"\n" + "="*50)
    print("PROCESSING ALL FRAMES FOR GIF")
    print("="*50)
    
    gif_frames = []
    all_frame_results = {}
    
    num_frames = len(frames) if isinstance(frames, list) else frames.shape[0]
    
    for frame_idx in range(num_frames):
        frame = frames[frame_idx] if isinstance(frames, list) else frames[frame_idx]
        
        vis_frame, segments_info = process_frame_with_sam(
            frame, mask_generator, tracks, visibility, outlier_set, frame_idx,
            min_outliers, outlier_ratio
        )
        
        # Create frame with big title instead of image annotations
        frame_with_title = create_frame_with_title(vis_frame, frame_idx, len(segments_info))
        gif_frames.append(frame_with_title)
        
        all_frame_results[frame_idx] = {
            'visualization': vis_frame,
            'segments_info': segments_info
        }
    
    # Create and save GIF
    print(f"\nCreating GIF with {len(gif_frames)} frames...")
    gif_path = os.path.join(output_dir, "sam_outlier_segments.gif")
    
    # Save as GIF with the PIL frames that already have titles
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=300,  # 300ms per frame
        loop=0,
        optimize=True,
        quality=95
    )
    
    print(f"GIF saved to: {gif_path}")
    
    # Print summary
    print(f"\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    total_kept_segments = sum(len(result['segments_info']) for result in all_frame_results.values())
    avg_segments_per_frame = total_kept_segments / num_frames
    
    print(f"Total frames processed: {num_frames}")
    print(f"Total segments kept across all frames: {total_kept_segments}")
    print(f"Average segments kept per frame: {avg_segments_per_frame:.1f}")
    print(f"Outlier criteria: ≥{min_outliers} outliers AND ≥{outlier_ratio*100}% outlier ratio")
    print(f"Results saved to: {output_dir}")
    
    return {
        'first_frame_result': {'visualization': first_vis, 'segments_info': first_info},
        'all_frame_results': all_frame_results,
        'gif_path': gif_path,
        'outlier_set': outlier_set,
        'criteria': {'min_outliers': min_outliers, 'outlier_ratio': outlier_ratio}
    }

# Example usage:
"""
# Assuming you have:
# - frames: your video frames
# - tracks: your track predictions  
# - visibility: your visibility predictions
# - outlier_tracks: list where outlier_tracks[0] contains the outlier track IDs

outlier_set = outlier_tracks[0]  # e.g., {1284, 1460, 1481, 1683, 1684}

results = run_sam_outlier_analysis(
    frames=frames,
    tracks=tracks, 
    visibility=visibility,
    outlier_tracks_level0=outlier_set,
    sam_checkpoint_path="sam_vit_b_01ec64.pth",  # Uses your existing vit_b checkpoint
    min_outliers=10,
    outlier_ratio=0.5,
    output_dir="sam_outlier_results"
)

# Or even simpler since sam_vit_b_01ec64.pth is now the default:
results = run_sam_outlier_analysis(
    frames=frames,
    tracks=tracks, 
    visibility=visibility,
    outlier_tracks_level0=outlier_tracks[0]
)
"""