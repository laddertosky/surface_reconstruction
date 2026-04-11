import threading

import numpy as np
import open3d as o3d


def AlphaShapeMethod(pcd: o3d.geometry.PointCloud, alpha: float, **kwargs) -> o3d.geometry.TriangleMesh:
    generated_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha, **kwargs)
    print(f"Alpha Shape method completed in {threading.current_thread()}")
    return generated_mesh

def BallPivotingMethod(pcd: o3d.geometry.PointCloud, radii: o3d.utility.DoubleVector, **kwargs) -> o3d.geometry.TriangleMesh:
    generated_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii, **kwargs)
    print(f"Ball Pivoting method completed in {threading.current_thread()}")
    return generated_mesh

def PoissonMethod(pcd: o3d.geometry.PointCloud, **kwargs) -> o3d.geometry.TriangleMesh:
    generated_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, **kwargs)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    generated_mesh.remove_vertices_by_mask(vertices_to_remove)

    print(f"Possion method completed in {threading.current_thread()}")
    return generated_mesh
