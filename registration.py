"""
Point cloud registration: FPFH + FGR + multi-scale ICP.

Pipeline:
  1. Preprocess: voxel downsample, normals, FPFH features
  2. FGR for coarse alignment
  3. Multi-scale point-to-plane ICP on downsampled clouds
"""
import numpy as np
import open3d as o3d

# Preprocessing
VOXEL_SIZE = 0.12
NORMAL_MAX_NN = 12
FPFH_MAX_NN = 35

# FGR
FGR_MAX_CORRESPONDENCE_FACTOR = 1.5
FGR_ITERATION_NUMBER = 12

# ICP
ICP_DISTANCES = (0.15, 0.05)
ICP_MAX_ITERATION = 12
ICP_RELATIVE_FITNESS = 1e-6
ICP_RELATIVE_RMSE = 1e-6


def _preprocess(pcd: o3d.geometry.PointCloud):
    """Downsample, estimate normals, compute FPFH."""
    down = pcd.voxel_down_sample(VOXEL_SIZE)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE * 2,
            max_nn=NORMAL_MAX_NN,
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=VOXEL_SIZE * 5,
            max_nn=FPFH_MAX_NN,
        ),
    )
    return down, fpfh


def register(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud,
) -> np.ndarray:
    """Return 4x4 transform aligning pcd2 to pcd1. main.py applies it to pcd2."""
    source = pcd2
    target = pcd1

    source_down, source_fpfh = _preprocess(source)
    target_down, target_fpfh = _preprocess(target)

    fgr_opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=VOXEL_SIZE * FGR_MAX_CORRESPONDENCE_FACTOR,
        iteration_number=FGR_ITERATION_NUMBER,
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, fgr_opt
    )
    trans = result.transformation

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=ICP_RELATIVE_FITNESS,
        relative_rmse=ICP_RELATIVE_RMSE,
        max_iteration=ICP_MAX_ITERATION,
    )
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    for max_dist in ICP_DISTANCES:
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, max_dist, trans, estimation, criteria
        )
        trans = result.transformation

    return trans
