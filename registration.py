import open3d as o3d
import numpy as np


def _preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """Downsample, estimate normals, and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
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
    This is the function that students will implement.
    It should take two open3d.geometry.PointCloud objects as input
    and return a 4x4 numpy array representing the transformation matrix
    that aligns pcd2 to pcd1.
    """
    # Step 1: source = pcd2 (we transform this), target = pcd1 (we align to this)
    source = pcd2
    target = pcd1

    # Step 2: Voxel downsampling (0.05 is a good default for indoor room-like scans)
    voxel_size = 0.05
    source_down, source_fpfh = _preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = _preprocess_point_cloud(target, voxel_size)

    # Step 5: Global registration with RANSAC
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999),
    )
    trans_init = result_ransac.transformation

    # Ensure full clouds have normals for point-to-plane ICP
    radius_normal = voxel_size * 2
    if not source.has_normals():
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

    # Step 6a: Point-to-point ICP for stable correction
    distance_threshold_icp = voxel_size * 0.4
    result_ptp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold_icp,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    # Step 6b: Point-to-plane ICP for tighter final alignment
    result_pt2pl = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold_icp,
        result_ptp.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )

    return result_pt2pl.transformation
