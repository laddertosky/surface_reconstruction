import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes

from Test import BuiltinSurfaceReconstructionMethod, BunnyPCD, ShowComparison

# =============================================================================
# Step 1: Set up a voxel grid around the input point cloud
# =============================================================================
# The original Poisson method is described with an adaptive octree.
# For this version, a regular voxel grid is used to build and test the
# main reconstruction pipeline. The grid is placed over the full bounding
# box of the point cloud with a small amount of extra space around it.
def _build_grid(points: np.ndarray, depth: int = 6, padding: float = 0.1):
    """
    Build a regular 3D voxel grid that covers the point cloud.

    Parameters
    ----------
    points : (N, 3) array
        Input point positions.
    depth : int
        Grid depth, where the number of cells per axis is 2^depth.
    padding : float
        Extra margin added around the bounding box.

    Returns
    -------
    grid_size : int
        Number of grid cells along each axis.
    grid_origin : (3,) array
        Minimum corner of the grid in world coordinates.
    voxel_size : float
        Side length of one voxel.
    """
    grid_size = 2 ** depth

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    max_extent = (bbox_max - bbox_min).max()
    padding_width = max_extent * padding

    grid_origin = bbox_min - padding_width
    voxel_size = (max_extent + 2 * padding_width) / grid_size

    return grid_size, grid_origin, voxel_size


# =============================================================================
# Step 2: Accumulate point normals onto the grid
# =============================================================================
# Each input point contributes its normal to nearby grid nodes.
# The full paper uses a smoother basis function, 
# For now, use trilinear splatting is a practical approximation.
def _splat_normals(points, normals, grid_size, grid_origin, voxel_size):
    """
    Spread point normals onto the voxel grid to form a vector field.

    Returns
    -------
    normal_field : (grid_size, grid_size, grid_size, 3) array
        Grid based vector field built from the input normals.
    """
    normal_field = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float64)
    # TODO:
    # Convert each point into fractional grid coordinates.
    # Use the surrounding 8 grid corners and distribute the normal
    # with trilinear weights.
    #
    # scaled_points = (points - grid_origin) / voxel_size
    # cell_index = floor(scaled_points)
    # local_offset = scaled_points - cell_index
    #
    # For each corner offset (dx, dy, dz) in {0, 1}:
    #   compute the trilinear weight
    #   add weight * normal to the matching grid node

    return normal_field


# =============================================================================
# Step 3: Compute the divergence of the vector field
# =============================================================================
# The divergence field becomes the right hand side of the Poisson solve.
# For now, finite differences on the voxel grid.
def _compute_divergence(normal_field, voxel_size):
    """
    Compute the divergence of the grid based vector field.

    Returns
    -------
    divergence_field : (grid_size, grid_size, grid_size) array
        Scalar field used as input to the Poisson equation.
    """
    # TODO:
    # Approximate the partial derivatives of each vector component.
    # One option is np.gradient.
    # Another option is to compute central differences directly.
    # dFx_dx = (F[i+1,j,k,0] - F[i-1,j,k,0]) / (2 * voxel_size)
    # dFy_dy = (F[i,j+1,k,1] - F[i,j-1,k,1]) / (2 * voxel_size)
    # dFz_dz = (F[i,j,k+1,2] - F[i,j,k-1,2]) / (2 * voxel_size)
    #
    # divergence_field = dFx_dx + dFy_dy + dFz_dz

    divergence_field = np.zeros(normal_field.shape[:3], dtype=np.float64)
    return divergence_field


# =============================================================================
# Step 4: Solve the Poisson equation on the grid
# =============================================================================
# This stage recovers a scalar field whose level set will define the surface.
# In grid form, this means solving a discrete Laplacian system.
def _solve_poisson(divergence_field, voxel_size):
    """
    Solve the Poisson equation on the voxel grid.

    Returns
    -------
    scalar_field : array with same shape as divergence_field
        Reconstructed scalar field used for surface extraction.
    """
    # TODO:
    # Two possible directions.
    # Option 1:
    # Build a sparse 3D Laplacian matrix and solve the linear system.
    # Option 2:
    # Use an FFT based method, which is often faster on a regular grid.
    # For the 3D Laplacian stencil, each grid cell is connected to its
    # 6 neighbors, with the center value balancing the sum.

    scalar_field = np.zeros_like(divergence_field)
    return scalar_field


# =============================================================================
# Step 5: Extract the mesh with marching cubes
# =============================================================================
# Once the scalar field is available, an isosurface can be extracted.
# Marching cubes converts that level set into a triangle mesh.
def _extract_isosurface(scalar_field, grid_origin, voxel_size, points):
    """
    Extract a triangle mesh from the reconstructed scalar field.

    Returns
    -------
    mesh : o3d.geometry.TriangleMesh
        Output mesh reconstructed from the scalar field.
    """
    # A common choice is to estimate the isovalue from scalar field values
    # near the input sample locations.
    # TODO: sample the scalar field at the input points and compute the mean
    iso_level = 0.0

    verts, faces, normals_mc, _ = marching_cubes(
        scalar_field,
        level=iso_level,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    verts = verts + grid_origin

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    return mesh


# =============================================================================
# Main reconstruction function
# =============================================================================
def PoissonMethod(pcd: o3d.geometry.PointCloud, depth: int = 6, **kwargs) -> o3d.geometry.TriangleMesh:
    """
    Reconstruct a mesh from an oriented point cloud using the Poisson pipeline.

    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input point cloud with estimated normals.
    depth : int
        Grid depth used to set the voxel resolution.
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Step 1: build the grid
    grid_size, grid_origin, voxel_size = _build_grid(points, depth=depth)

    # Step 2: place normals onto the grid
    normal_field = _splat_normals(points, normals, grid_size, grid_origin, voxel_size)

    # Step 3: compute divergence
    divergence_field = _compute_divergence(normal_field, voxel_size)

    # Step 4: solve for the scalar field
    scalar_field = _solve_poisson(divergence_field, voxel_size)

    # Step 5: extract the mesh
    mesh = _extract_isosurface(scalar_field, grid_origin, voxel_size, points)

    return mesh


if __name__ == "__main__":
    pcd = BunnyPCD()
    generated_mesh = PoissonMethod(pcd)
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.POISSON)