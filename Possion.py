import open3d as o3d

from Test import BuiltinSurfaceReconstructionMethod, BunnyPCD, ShowComparison


def PoissonMethod(pcd: o3d.geometry.PointCloud, **kwargs) -> o3d.geometry.TriangleMesh:
    # TODO:
    pass

if __name__ == "__main__":
    pcd = BunnyPCD()
    generated_mesh = PoissonMethod(pcd)
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.POISSON)
