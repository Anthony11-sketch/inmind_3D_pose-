"""
Point cloud registration: FPFH + FGR + multi-scale ICP.

Pipeline:
  1. Preprocess: voxel downsample, normals, FPFH features (parallel)
  2. FGR for coarse alignment (CPU)
  3. Multi-scale ICP refinement (GPU if enabled, else CPU)

Config: config.yaml (override with env CONFIG_USE_GPU=1 for GPU)
"""
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from concurrent.futures import ThreadPoolExecutor

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

def _load_cfg():
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    use_gpu = os.environ.get("CONFIG_USE_GPU", "").lower()
    if use_gpu:
        cfg["device"]["use_gpu"] = use_gpu in ("1", "true", "yes")
    return cfg


def _preprocess(pcd: o3d.geometry.PointCloud, cfg: dict):
    """Downsample, estimate normals, compute FPFH."""
    p = cfg["preprocessing"]
    voxel = p["voxel_size"]
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * p["normal_radius_factor"],
            max_nn=p["normal_max_nn"],
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * p["fpfh_radius_factor"],
            max_nn=p["fpfh_max_nn"],
        ),
    )
    return down, fpfh


def _downsample_with_normals(pcd: o3d.geometry.PointCloud, voxel: float, cfg: dict):
    """Downsample and estimate normals at given voxel size."""
    nn = cfg["preprocessing"]["normal_max_nn"]
    r = cfg["preprocessing"]["normal_radius_factor"]
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * r, max_nn=nn)
    )
    return down


def _refine_icp_cpu(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    trans: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """CPU refinement: multi-pass ICP on downsampled + final refinement."""
    icp = cfg["icp"]
    ref = cfg["icp_refinement"]
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=icp["relative_fitness"],
        relative_rmse=icp["relative_rmse"],
        max_iteration=icp["max_iteration"],
    )
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    for max_dist in icp["distances"]:
        result = o3d.pipelines.registration.registration_icp(
            source, target, max_dist, trans, estimation, criteria
        )
        trans = result.transformation

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_s = ex.submit(_downsample_with_normals, source, ref["voxel"], cfg)
        fut_t = ex.submit(_downsample_with_normals, target, ref["voxel"], cfg)
        source_ref, target_ref = fut_s.result(), fut_t.result()
    criteria_ref = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=icp["relative_fitness"],
        relative_rmse=icp["relative_rmse"],
        max_iteration=ref["iter"],
    )
    result = o3d.pipelines.registration.registration_icp(
        source_ref, target_ref, ref["dist"], trans, estimation, criteria_ref
    )
    return np.asarray(result.transformation)


def _refine_icp_gpu(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    trans: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """GPU refinement: multi-scale ICP on CUDA using Tensor API."""
    treg = o3d.t.pipelines.registration
    dev = cfg["device"]
    ms = cfg["icp_multiscale"]
    icp = cfg["icp"]

    source_t = o3d.t.geometry.PointCloud.from_legacy(source)
    target_t = o3d.t.geometry.PointCloud.from_legacy(target)
    source_t = source_t.cuda(dev["gpu_device_id"])
    target_t = target_t.cuda(dev["gpu_device_id"])

    init = o3d.core.Tensor(trans, dtype=o3d.core.Dtype.Float64)
    voxel_sizes = o3d.utility.DoubleVector(ms["voxel_sizes"])
    max_distances = o3d.utility.DoubleVector(ms["max_correspondence_distances"])
    criteria_list = [
        treg.ICPConvergenceCriteria(
            relative_fitness=icp["relative_fitness"],
            relative_rmse=icp["relative_rmse"],
            max_iteration=n,
        )
        for n in ms["criteria_max_iterations"]
    ]
    estimation = treg.TransformationEstimationPointToPlane()

    result = treg.multi_scale_icp(
        source_t, target_t, voxel_sizes, criteria_list,
        max_distances, init, estimation,
    )
    t_cpu = result.transformation.cpu()
    return t_cpu.numpy() if hasattr(t_cpu, "numpy") else np.asarray(t_cpu)


def register(
    pcd1: o3d.geometry.PointCloud,
    pcd2: o3d.geometry.PointCloud,
) -> np.ndarray:
    """Return 4x4 transform aligning pcd2 to pcd1. main.py applies it to pcd2."""
    cfg = _load_cfg()

    source = pcd2
    target = pcd1

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_s = ex.submit(_preprocess, source, cfg)
        fut_t = ex.submit(_preprocess, target, cfg)
        source_down, source_fpfh = fut_s.result()
        target_down, target_fpfh = fut_t.result()

    p = cfg["preprocessing"]
    fgr = cfg["fgr"]
    fgr_opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=p["voxel_size"] * fgr["max_correspondence_factor"],
        iteration_number=fgr["iteration_number"],
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, fgr_opt
    )
    trans = np.asarray(result.transformation)

    if cfg["device"]["use_gpu"]:
        trans = _refine_icp_gpu(source, target, trans, cfg)
    else:
        trans = _refine_icp_cpu(source_down, target_down, trans, cfg)

    return trans
