import numpy as np
import torch

def get_errors(all_results, error_type = None):
    inlier_tracks = (all_results[('track_based', 'frame_zero_to_all')][2]['frame_pair_details']['global_inlier_tracks'])
    outlier_tracks = (all_results[('track_based', 'frame_zero_to_all')][2]['frame_pair_details']['global_outlier_tracks'])
    errors = all_results[('track_based', 'frame_zero_to_all')][2]['frame_pair_details']['global_track_errors']

    errors_inlier = np.sum(list({tid: errors[tid] for tid in inlier_tracks if tid in errors}.values()))
    errors_outlier = np.sum(list({tid: errors[tid] for tid in outlier_tracks if tid in errors}.values()))

    errors_total = errors_inlier + errors_outlier

    if error_type is None:
        return errors_inlier, errors_outlier, errors_total
    elif error_type == 'inlier':
        return list({tid: errors[tid] for tid in inlier_tracks if tid in errors}.values())
    elif error_type == 'outlier':
        return list({tid: errors[tid] for tid in outlier_tracks if tid in errors}.values())
    else:
        return None
    
def get_inlier_tracks(all_results):
    return (all_results[('track_based', 'frame_zero_to_all')][2]['frame_pair_details']['global_inlier_tracks'])

def get_outlier_tracks(all_results):
    return (all_results[('track_based', 'frame_zero_to_all')][2]['frame_pair_details']['global_outlier_tracks'])


def get_3d_coordinates_for_frame(frames, model, device, frame_idx):
    """Get 3D coordinates and intrinsics for a specific frame using MoGe"""
    input_image = frames[frame_idx]
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    
    output = model.infer(input_image)
    return output["points"], output["mask"], output["intrinsics"]
