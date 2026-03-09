"""
Point cloud registration: FPFH + FGR + multi-scale ICP.

Pipeline:
  1. Preprocess: voxel downsample, normals, FPFH features (parallel)
  2. FGR for coarse alignment
  3. Multi-scale point-to-plane ICP on downsampled clouds
"""
import numpy as np
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor

# Preprocessing
VOXEL_SIZE = 0.17
NORMAL_MAX_NN = 8
FPFH_MAX_NN = 25

# FGR
FGR_MAX_CORRESPONDENCE_FACTOR = 1.5
FGR_ITERATION_NUMBER = 5

# ICP
ICP_DISTANCES = (0.15, 0.05)
ICP_MAX_ITERATION = 5
ICP_RELATIVE_FITNESS = 1e-6
ICP_RELATIVE_RMSE = 1e-6
# Final refinement (no full-cloud ICP for speed)
ICP_REFINEMENT_VOXEL = 0.08
ICP_REFINEMENT_DIST = 0.02
ICP_REFINEMENT_ITER = 2


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
            radius=VOXEL_SIZE * 4,
            max_nn=FPFH_MAX_NN,
        ),
    )
    return down, fpfh


def _downsample_with_normals(pcd: o3d.geometry.PointCloud, voxel: float):
    """Downsample and estimate normals at given voxel size."""
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=NORMAL_MAX_NN)
    )
    return down


def register(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud,
) -> np.ndarray:
    """Return 4x4 transform aligning pcd2 to pcd1. main.py applies it to pcd2."""
    source = pcd2
    target = pcd1

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_s = ex.submit(_preprocess, source)
        fut_t = ex.submit(_preprocess, target)
        source_down, source_fpfh = fut_s.result()
        target_down, target_fpfh = fut_t.result()

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

    # Final refinement on denser clouds with 0.02 threshold
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_s = ex.submit(_downsample_with_normals, source, ICP_REFINEMENT_VOXEL)
        fut_t = ex.submit(_downsample_with_normals, target, ICP_REFINEMENT_VOXEL)
        source_ref, target_ref = fut_s.result(), fut_t.result()
    criteria_ref = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=ICP_RELATIVE_FITNESS,
        relative_rmse=ICP_RELATIVE_RMSE,
        max_iteration=ICP_REFINEMENT_ITER,
    )
    result = o3d.pipelines.registration.registration_icp(
        source_ref, target_ref, ICP_REFINEMENT_DIST, trans, estimation, criteria_ref
    )
    trans = result.transformation

    return trans
