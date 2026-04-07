"""
# NOTE:
# This code was generated with assistance from ChatGPT (OpenAI).
# It has been reviewed and adapted as needed for this project.

Usage: see the arguments in `show_comparison` function and `BuiltinSurfaceReconstructionMethod` enum
ShowComparison(...) will print the statistic content defined in `MeshCompareResult`

How to read the result:

Smaller chamfer_mean, chamfer_rms, chamfer_p95, and hausdorff means your mesh is closer to the built-in output. compute_point_cloud_distance() measures nearest-point distances from one sampled surface to the other, so the symmetric version is a good surface-to-surface comparison.

Higher precision, recall, and fscore mean more overlap within tolerance tau.
Topology flags help you detect whether the implementation matches the built-in mesh quality properties, not just geometry. Open3D explicitly exposes these checks.

For a developer-facing report, the most useful single number is usually fscore at a chosen tolerance, plus chamfer_mean and hausdorff.
"""
import copy
import json
from dataclasses import asdict, dataclass
from enum import Enum, auto

import numpy as np
import open3d as o3d


class BuiltinSurfaceReconstructionMethod(Enum):
    POISSON = auto()
    BALL_PIVOTING = auto()
    ALPHA_SHAPE = auto()

@dataclass
class MeshCompareResult:
    ## Smaller is better
    chamfer_mean: float
    chamfer_rms: float
    chamfer_p95: float
    hausdorff: float

    ## Higher is better
    precision: float
    recall: float
    fscore: float


def _to_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh = copy.deepcopy(mesh)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def _sample_mesh(mesh: o3d.geometry.TriangleMesh, n_samples: int):
    return _to_mesh(mesh).sample_points_uniformly(number_of_points=n_samples)


def _compare_candidate_to_builtin(
    pcd: o3d.geometry.PointCloud,
    candidate_mesh: o3d.geometry.TriangleMesh,
    method: BuiltinSurfaceReconstructionMethod = BuiltinSurfaceReconstructionMethod.POISSON,
    n_samples: int = 50000,
    tau: float | None = None,
    **builtin_kwargs,
) -> MeshCompareResult:

    pcd = copy.deepcopy(pcd)
    if not pcd.has_normals():
        raise ValueError("Point cloud must have normals.")

    if not isinstance(method, BuiltinSurfaceReconstructionMethod):
        raise TypeError(f"method must be a BuiltinMethod, got {type(method).__name__}")

    # --- enum-based dispatch ---
    if method is BuiltinSurfaceReconstructionMethod.POISSON:
        ref_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, **builtin_kwargs
        )
    elif method is BuiltinSurfaceReconstructionMethod.BALL_PIVOTING:
        radii = builtin_kwargs.get("radii")
        ref_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, radii
        )
    elif method is BuiltinSurfaceReconstructionMethod.ALPHA_SHAPE:
        alpha = builtin_kwargs.get("alpha")
        ref_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha
        )

    ref_mesh = _to_mesh(ref_mesh)
    candidate_mesh = _to_mesh(candidate_mesh)

    ref_pc = _sample_mesh(ref_mesh, n_samples)
    cand_pc = _sample_mesh(candidate_mesh, n_samples)

    d_ref_to_cand = np.asarray(ref_pc.compute_point_cloud_distance(cand_pc))
    d_cand_to_ref = np.asarray(cand_pc.compute_point_cloud_distance(ref_pc))

    chamfer_mean = float(0.5 * (d_ref_to_cand.mean() + d_cand_to_ref.mean()))
    chamfer_rms = float(
        0.5 * (np.sqrt(np.mean(d_ref_to_cand**2)) + np.sqrt(np.mean(d_cand_to_ref**2)))
    )
    chamfer_p95 = float(
        0.5 * (np.percentile(d_ref_to_cand, 95) + np.percentile(d_cand_to_ref, 95))
    )
    hausdorff = float(max(d_ref_to_cand.max(), d_cand_to_ref.max()))

    if tau is None:
        nn = np.asarray(pcd.compute_nearest_neighbor_distance())
        tau = float(np.median(nn)) if len(nn) else 0.0

    precision = float(np.mean(d_cand_to_ref <= tau))
    recall = float(np.mean(d_ref_to_cand <= tau))
    fscore = float(
        0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    )

    return MeshCompareResult(
        chamfer_mean=chamfer_mean,
        chamfer_rms=chamfer_rms,
        chamfer_p95=chamfer_p95,
        hausdorff=hausdorff,
        precision=precision,
        recall=recall,
        fscore=fscore,
    )

def ShowComparison(
    pcd: o3d.geometry.PointCloud,
    candidate_mesh: o3d.geometry.TriangleMesh,
    method: BuiltinSurfaceReconstructionMethod = BuiltinSurfaceReconstructionMethod.POISSON,
    n_samples: int = 50000,
    tau: float | None = None,
    **builtin_kwargs,
) -> None:
    
    result = _compare_candidate_to_builtin(pcd, candidate_mesh, method, n_samples, tau, **builtin_kwargs)
    print(json.dumps(asdict(result), indent=2))

def BunnyPCD() -> o3d.geometry.PointCloud:
    """
    A point cloud for testing purpose.
    """
    mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)

    o3d.utility.random.seed(0)
    pcd = mesh.sample_points_poisson_disk(100)
    pcd.estimate_normals()
    return pcd
