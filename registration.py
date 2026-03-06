"""
Global registration pipeline (improved for accuracy):

1. Preprocess: voxel downsample, estimate normals, compute FPFH
2. RANSAC on downsampled clouds
3. Multi-scale ICP: coarse on downsampled, then fine on full clouds
   - ICP pass 1: downsampled, threshold 0.08
   - ICP pass 2: downsampled, threshold 0.04
   - ICP pass 3: full clouds, threshold 0.02
"""
import open3d as o3d
import numpy as np

VOXEL_SIZE = 0.04  # Smaller = finer, better accuracy (0.03–0.06)
NORMAL_RADIUS_FACTOR = 2.0  # For preprocess
NORMAL_RADIUS_FULL = 3.0    # For full clouds: more careful normals
NORMAL_MAX_NN = 50          # For full clouds


def _preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """Downsample, estimate normals, compute FPFH."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * NORMAL_RADIUS_FACTOR
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def register(pcd1: o3d.geometry.PointCloud, pcd2: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Returns 4x4 transformation that aligns pcd1 to pcd2 (apply to pcd1).
    """
    source = pcd1
    target = pcd2
    voxel_size = VOXEL_SIZE

    # 1. Preprocess
    source_down, source_fpfh = _preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = _preprocess_point_cloud(target, voxel_size)

    # 2. RANSAC on downsampled clouds
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    trans = result_ransac.transformation

    # 3. Multi-scale ICP: coarse on downsampled, then fine on full clouds
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
    )

    # ICP pass 1 & 2: downsampled clouds (coarse, medium)
    for dist in (0.08, 0.04):
        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            dist,
            trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            icp_criteria,
        )
        trans = result.transformation

    # ICP pass 3: full clouds with finer threshold
    if not source.has_normals():
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * NORMAL_RADIUS_FULL, max_nn=NORMAL_MAX_NN
            )
        )
    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * NORMAL_RADIUS_FULL, max_nn=NORMAL_MAX_NN
            )
        )
    result_final = o3d.pipelines.registration.registration_icp(
        source,
        target,
        0.02,  # fine threshold
        trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        icp_criteria,
    )

    return result_final.transformation
