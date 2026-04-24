"""
Poisson Surface Reconstruction
Based on:
    Kazhdan, M., Bolitho, M., & Hoppe, H. (2006).
    "Poisson Surface Reconstruction."
    Eurographics Symposium on Geometry Processing.
    https://hhoppe.com/poissonrecon.pdf
"""

import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes

from Test import BuiltinSurfaceReconstructionMethod, BunnyPCD, ShowComparison

# =============================================================================
# Step 1: Set up a voxel grid around the input point cloud
# =============================================================================
# Chop space into a 3D grid of little cubes.
# Grid is a bit bigger than the object with extra space so the surface never touches the grid boundary.

# The original Poisson method is described with an adaptive octree.
# For this version, a regular voxel grid is used to build and test the
# main reconstruction pipeline. 
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
        Side length of one voxel (cube) in world units.
        a voxel_size of 0.5 would mean the grid_corner (2,0,0) sits at
        world position (1,0,0).
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
# Every normal is floating at a random point, grid wants to have them at grid corners.
# For each arrow, find 8 corners of the little cube that it is inside, 
# and split 'vote' among the 8 corners based on how close it is to each one.
# Point in the middle gives each corner 1/8 of the vote, 
# Point closer to one corner gives that corner almost everything. 

# Each input point contributes its normal to nearby grid nodes.
# The full paper uses a smoother basis function, 
# We use trilinear splatting, as a simpler smooth kernel
def _splat_normals(points, normals, grid_size, grid_origin, voxel_size):
    """
    Spread point normals onto the voxel grid to form a vector field V. 
    
    Each input normal is distributed over the 8 grid corners of the cell
    that contains the point, using trilinear weights.

    Returns
    -------
    normal_field : (grid_size, grid_size, grid_size, 3) array
        Grid based vector field built from the input normals.
    """
    normal_field = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float64)
    
    # convert each point into fractional grid coordinates.
    # For our example point p = (1.3, 2.8, 0.1) with origin (0,0,0):
    # scaled = ((1.3, 2.8, 0.1) - (0, 0, 0)) / 0.5 = (2.6, 5.6, 0.2)
    # meaning the point is 2.6 grid cells over in x, 5.6 grid cells up in y, 0.2 grid cells deep in z.
    scaled_points = (points - grid_origin) / voxel_size #(N, 3)

    # Integer part is index of 'low' corner of the containing cell,
    # says which grid cell contains the point.
    # cell_index[0] is a cell that contains point 0, cell_index[1] contains point 1...
    # we find the cell for every point in the point cloud
    # cell_index stores the 'low corner' of the cell containing the point.
    cell_index = np.floor(scaled_points).astype(np.int64) #(N,3)

    # Fractional part is the local offset within the cell
    # says where inside the grid cell the point is.
    # local_offset[i] = 0 --> point sits on low corner along the axis
    # local_offset[i] = 1 --> point sits on high corner
    local_offset = scaled_points - cell_index #(N,3)

    # clip so cell_index+1 never exceeds grid_size-1
    # (padding in step 1 already should prevent this)
    cell_index = np.clip(cell_index, 0, grid_size - 2)

    # for each of 8 corners (dx, dy, dz) in {0,1}^3, accumulate 
    # the trilinear-weighted normal at the corner.
    # (basically for the corners ranging from (0,0,0), (1,0,0),...,(1,1,1))
    for dx in (0,1):
        for dy in (0,1):
            for dz in (0,1):
                # trilinear weight for this corner
                # if dx=0, weight = (1-local_offset_x). if dx=1, want weight = local_offset_x
                
                # low x corner ---------------- high x corner
                #         point is here (local_offset_x = 0.2)
                #         0.2 from low, 0.8 from high,
                # weight for low x side  = 1 - 0.2 = 0.8
                # weight for high x side = 0.2
                
                # example: if a point has local offset: [0.2,, 0.7, 0.4]
                # for the corner (0,1,0) --> weight is: (1-0.2) * (0.7) * (1-0.4)
                # for the corner (1,1,0) --> weight is (0.2) * (0.7) * (1-0.4)
                wx = local_offset[:,0] if dx == 1 else (1.0 - local_offset[:,0])
                wy = local_offset[:,1] if dy == 1 else (1.0 - local_offset[:,1])
                wz = local_offset[:,2] if dz == 1 else (1.0 - local_offset[:,2])
                weights = (wx * wy * wz)[:, None] #(N,1), it broadcasts over 3 normal components

                ix = cell_index[:,0] + dx
                iy = cell_index[:,1] + dy
                iz = cell_index[:,2] + dz

                # np.add.at handles repeated indices
                np.add.at(normal_field, (ix, iy, iz), weights * normals)

    return normal_field


# =============================================================================
# Step 3: Compute the divergence of the vector field
# =============================================================================
# At each grid cell, look at arrows nearby, and where they point
# point away: positive divergence. point toward: negative. canecl out: zero
# This gives a single number per cell (right hand side of the poisson equation)
# Mathematically: div V = dVx/dx + dVy/dy + dVz/dz.
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
    # np.gradient takes the derivative along a given axis with spacing = voxel_size.
    # We take dVx/dx, dVy/dy, dVz/dz and sum.
    dVx_dx = np.gradient(normal_field[..., 0], voxel_size, axis=0)
    dVy_dy = np.gradient(normal_field[..., 1], voxel_size, axis=1)
    dVz_dz = np.gradient(normal_field[..., 2], voxel_size, axis=2)
    
    divergence_field = dVx_dx + dVy_dy + dVz_dz
    return divergence_field


# =============================================================================
# Step 4: Solve the Poisson equation on the grid
# =============================================================================
# We have "how insideness is changing" at every cell (divergence),
# and we want the actual insideness values. This is the Poisson
# equation,  Laplacian(chi) = div(V).
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