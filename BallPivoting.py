import open3d as o3d

from Test import BuiltinSurfaceReconstructionMethod, BunnyPCD, ShowComparison


def BallPivotingMethod(pcd: o3d.geometry.PointCloud, radii: o3d.utility.DoubleVector, **kwargs) -> o3d.geometry.TriangleMesh:
    # TODO:
    pass

if __name__ == "__main__":
    pcd = BunnyPCD()
    radii = o3d.utility.DoubleVector([0.005, 0.01, 0.02])
    ref_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
    generated_mesh = BallPivotingMethod(pcd, radii)
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.BALL_PIVOTING, radii=radii)
