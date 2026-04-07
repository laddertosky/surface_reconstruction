"""

This file is used to verify if code written in Test.py can differentiate the meshes properly.
Usually you don't need to use this one.

"""
import open3d as o3d

from Test import BuiltinSurfaceReconstructionMethod, BunnyPCD, ShowComparison


def verify_poisson_built_in():
    """
    Should match exactly with builtin poisson method
    """
    pcd = BunnyPCD()
    generated_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.POISSON)

def verify_alpha_shape_built_in():
    """
    Should match exactly with builtin alpha shape method
    """
    pcd = BunnyPCD()
    generated_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.5)
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.ALPHA_SHAPE, alpha=0.5)

def verify_ball_pivoting_built_in():
    """
    Should match exactly with builtin ball pivoting method
    """
    pcd = BunnyPCD()
    radii = o3d.utility.DoubleVector([0.005, 0.01, 0.04])
    generated_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.BALL_PIVOTING, radii=radii)

def verify_poisson_vs_ball_pivoting():
    """
    Should show difference when two meshes are generated differently
    """
    pcd = BunnyPCD()
    generated_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
    radii = o3d.utility.DoubleVector([0.005, 0.01, 0.04])
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.BALL_PIVOTING, radii=radii)

if __name__ == "__main__":
    verify_poisson_built_in()
    verify_ball_pivoting_built_in()
    verify_alpha_shape_built_in()
    verify_poisson_vs_ball_pivoting()
