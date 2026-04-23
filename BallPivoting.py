"""
Ball Pivoting Algorithm (BPA) surface reconstruction.

This implementation follows the core serial BPA described in:
    Bernardini et al. (1999), "The Ball-Pivoting Algorithm for Surface Reconstruction"
    Digne (2014), "An Analysis and Implementation of a Parallel Ball Pivoting Algorithm"

The important algorithmic points preserved here are:
    - seeds are searched only among orphan vertices
    - front expansion pivots around oriented front edges
    - edges that cannot pivot become boundary edges
    - multi-radius BPA reactivates boundary edges for the next radius instead of reseeding
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from Test import BuiltinSurfaceReconstructionMethod, ShowComparison

GEOM_EPS = 1e-10
NORMAL_EPS = 1e-6


@dataclass
class Edge:
    i: int
    j: int
    opposite: int
    ball_center: np.ndarray


def _triangle_geometry(pi, pj, pk):
    """Return circumcenter, oriented plane normal, and circumradius^2."""
    v0 = pj - pi
    v1 = pk - pi

    normal = np.cross(v0, v1)
    norm_len = np.linalg.norm(normal)
    if norm_len < GEOM_EPS:
        return None
    normal = normal / norm_len

    d00 = float(np.dot(v0, v0))
    d11 = float(np.dot(v1, v1))
    d01 = float(np.dot(v0, v1))
    denom = 2.0 * (d00 * d11 - d01 * d01)
    if abs(denom) < GEOM_EPS:
        return None

    s = d11 * (d00 - d01) / denom
    t = d00 * (d11 - d01) / denom
    circumcenter = pi + s * v0 + t * v1
    circumradius_sq = float(np.sum((circumcenter - pi) ** 2))
    return circumcenter, normal, circumradius_sq


def _ball_center_from_normal(pi, pj, pk, rho, normal_hint):
    """Center of the rho-ball on the side indicated by normal_hint."""
    geometry = _triangle_geometry(pi, pj, pk)
    if geometry is None:
        return None

    circumcenter, plane_normal, circumradius_sq = geometry
    gap = rho * rho - circumradius_sq
    if gap < -GEOM_EPS:
        return None

    if np.dot(plane_normal, normal_hint) < 0.0:
        plane_normal = -plane_normal

    lift = np.sqrt(max(gap, 0.0))
    return circumcenter + lift * plane_normal


def _ball_center_from_reference(pi, pj, pk, rho, reference_center):
    """
    Center of the rho-ball on the same side of the triangle plane as reference_center.

    This is used when a boundary edge is reactivated for a larger radius.
    """
    geometry = _triangle_geometry(pi, pj, pk)
    if geometry is None:
        return None

    circumcenter, plane_normal, circumradius_sq = geometry
    gap = rho * rho - circumradius_sq
    if gap < -GEOM_EPS:
        return None

    if np.dot(reference_center - circumcenter, plane_normal) < 0.0:
        plane_normal = -plane_normal

    lift = np.sqrt(max(gap, 0.0))
    return circumcenter + lift * plane_normal


def _triangle_unit_normal(points, tri):
    a, b, c = tri
    normal = np.cross(points[b] - points[a], points[c] - points[a])
    norm_len = np.linalg.norm(normal)
    if norm_len < GEOM_EPS:
        return None
    return normal / norm_len


def _triangle_is_compatible(points, normals, tri):
    """
    A triangle is compatible when its normal has positive dot product
    with the normals of its three vertices.
    """
    tri_normal = _triangle_unit_normal(points, tri)
    if tri_normal is None:
        return False, None

    dots = normals[np.asarray(tri)] @ tri_normal
    if np.min(dots) < -NORMAL_EPS:
        return False, None
    return True, tri_normal


def _orient_seed_triangle(points, normals, i, j, k):
    tri = (i, j, k)
    compatible, tri_normal = _triangle_is_compatible(points, normals, tri)
    if compatible:
        return tri, tri_normal

    tri = (i, k, j)
    compatible, tri_normal = _triangle_is_compatible(points, normals, tri)
    if compatible:
        return tri, tri_normal

    return None, None


def _ball_is_empty(center, rho, points, kdtree, skip_indices):
    eps = max(GEOM_EPS, 1e-8 * rho)
    nearby = kdtree.query_ball_point(center, rho - eps)
    skip = set(skip_indices)
    for idx in nearby:
        if idx in skip:
            continue
        if np.linalg.norm(points[idx] - center) < rho - eps:
            return False
    return True


def _is_inner_vertex(vertex_used, open_edge_count, idx):
    return bool(vertex_used[idx] and open_edge_count[idx] == 0)


def _find_seed_triangle(points, normals, kdtree, rho, vertex_used):
    """Find a valid seed triangle among orphan vertices only."""
    n_points = len(points)
    search_radius = 2.0 * rho

    for i in range(n_points):
        if vertex_used[i]:
            continue

        neighbors = [
            idx
            for idx in kdtree.query_ball_point(points[i], search_radius)
            if idx != i and not vertex_used[idx]
        ]
        neighbors.sort(key=lambda idx: float(np.linalg.norm(points[idx] - points[i])))

        for offset, j in enumerate(neighbors):
            for k in neighbors[offset + 1 :]:
                seed_tri, tri_normal = _orient_seed_triangle(points, normals, i, j, k)
                if seed_tri is None:
                    continue

                center = _ball_center_from_normal(
                    points[seed_tri[0]],
                    points[seed_tri[1]],
                    points[seed_tri[2]],
                    rho,
                    tri_normal,
                )
                if center is None:
                    continue

                if _ball_is_empty(center, rho, points, kdtree, seed_tri):
                    return seed_tri, center

    return None


def _pivot(points, normals, kdtree, rho, edge, vertex_used, open_edge_count):
    """Roll the ball around a front edge and return the first valid hit."""
    pi = points[edge.i]
    pj = points[edge.j]
    midpoint = 0.5 * (pi + pj)

    edge_vec = pj - pi
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < GEOM_EPS:
        return None
    edge_axis = edge_vec / edge_len

    old_rel = edge.ball_center - midpoint
    old_proj = old_rel - np.dot(old_rel, edge_axis) * edge_axis
    old_proj_len = np.linalg.norm(old_proj)
    if old_proj_len < GEOM_EPS:
        return None
    old_proj_unit = old_proj / old_proj_len
    perp = np.cross(edge_axis, old_proj_unit)

    candidates = []
    for k in kdtree.query_ball_point(midpoint, 2.0 * rho):
        if k in (edge.i, edge.j, edge.opposite):
            continue
        if _is_inner_vertex(vertex_used, open_edge_count, k):
            continue

        tri = (edge.j, edge.i, k)
        compatible, tri_normal = _triangle_is_compatible(points, normals, tri)
        if not compatible:
            continue

        center = _ball_center_from_normal(
            points[tri[0]],
            points[tri[1]],
            points[tri[2]],
            rho,
            tri_normal,
        )
        if center is None:
            continue

        new_rel = center - midpoint
        new_proj = new_rel - np.dot(new_rel, edge_axis) * edge_axis
        new_proj_len = np.linalg.norm(new_proj)
        if new_proj_len < GEOM_EPS:
            continue
        new_proj_unit = new_proj / new_proj_len

        cos_angle = np.clip(np.dot(old_proj_unit, new_proj_unit), -1.0, 1.0)
        sin_angle = np.dot(perp, new_proj_unit)
        angle = np.arctan2(sin_angle, cos_angle)
        if angle <= GEOM_EPS:
            angle += 2.0 * np.pi

        candidates.append((angle, k, center))

    candidates.sort(key=lambda item: item[0])
    for _, k, center in candidates:
        if _ball_is_empty(center, rho, points, kdtree, (edge.i, edge.j, k)):
            return k, center

    return None


def _add_triangle(tri, triangles, seen_triangles, used_dir_edges):
    """
    Add an oriented triangle if it keeps the mesh orientable and edge-manifold.

    Each undirected edge can appear at most twice, once in each direction.
    """
    key = tuple(sorted(tri))
    if key in seen_triangles:
        return False

    dir_edges = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    if any(edge in used_dir_edges for edge in dir_edges):
        return False

    seen_triangles.add(key)
    used_dir_edges.update(dir_edges)
    triangles.append(tri)
    return True


def _add_open_edge(edge, container, open_edge_count):
    key = (edge.i, edge.j)
    if key in container:
        return

    container[key] = edge
    open_edge_count[edge.i] += 1
    open_edge_count[edge.j] += 1


def _remove_open_edge(key, container, open_edge_count):
    edge = container.pop(key, None)
    if edge is None:
        return None

    open_edge_count[edge.i] -= 1
    open_edge_count[edge.j] -= 1
    return edge


def _move_front_edge_to_boundary(edge, front_edges, boundary_edges):
    removed = front_edges.pop((edge.i, edge.j), None)
    if removed is not None:
        boundary_edges[(edge.i, edge.j)] = edge


def _pop_matching_open_edge(key, front_edges, boundary_edges, open_edge_count):
    edge = _remove_open_edge(key, front_edges, open_edge_count)
    if edge is not None:
        return edge
    return _remove_open_edge(key, boundary_edges, open_edge_count)


def _expand_front(
    points,
    normals,
    kdtree,
    rho,
    front_queue,
    front_edges,
    boundary_edges,
    vertex_used,
    open_edge_count,
    triangles,
    seen_triangles,
    used_dir_edges,
):
    triangles_before = len(triangles)

    while front_queue:
        edge = front_queue.popleft()
        if (edge.i, edge.j) not in front_edges:
            continue

        result = _pivot(points, normals, kdtree, rho, edge, vertex_used, open_edge_count)
        if result is None:
            _move_front_edge_to_boundary(edge, front_edges, boundary_edges)
            continue

        new_k, new_center = result
        new_tri = (edge.j, edge.i, new_k)
        if not _add_triangle(new_tri, triangles, seen_triangles, used_dir_edges):
            _move_front_edge_to_boundary(edge, front_edges, boundary_edges)
            continue

        _remove_open_edge((edge.i, edge.j), front_edges, open_edge_count)
        vertex_used[new_k] = True

        new_edges = (
            Edge(edge.i, new_k, edge.j, new_center),
            Edge(new_k, edge.j, edge.i, new_center),
        )
        for new_edge in new_edges:
            reverse_key = (new_edge.j, new_edge.i)
            if _pop_matching_open_edge(reverse_key, front_edges, boundary_edges, open_edge_count) is None:
                _add_open_edge(new_edge, front_edges, open_edge_count)
                front_queue.append(new_edge)

    return len(triangles) - triangles_before


def _grow_from_seed(
    points,
    normals,
    kdtree,
    rho,
    seed_tri,
    seed_center,
    vertex_used,
    open_edge_count,
    boundary_edges,
    triangles,
    seen_triangles,
    used_dir_edges,
):
    if not _add_triangle(seed_tri, triangles, seen_triangles, used_dir_edges):
        return 0

    for idx in seed_tri:
        vertex_used[idx] = True

    front_edges = {}
    front_queue = deque()
    seed_edges = (
        Edge(seed_tri[0], seed_tri[1], seed_tri[2], seed_center),
        Edge(seed_tri[1], seed_tri[2], seed_tri[0], seed_center),
        Edge(seed_tri[2], seed_tri[0], seed_tri[1], seed_center),
    )
    for edge in seed_edges:
        _add_open_edge(edge, front_edges, open_edge_count)
        front_queue.append(edge)

    return 1 + _expand_front(
        points,
        normals,
        kdtree,
        rho,
        front_queue,
        front_edges,
        boundary_edges,
        vertex_used,
        open_edge_count,
        triangles,
        seen_triangles,
        used_dir_edges,
    )


def _reactivate_boundary_edges(points, kdtree, rho, boundary_edges):
    """
    Multi-radius BPA step from Bernardini/Digne:
    do not reseed, only reactivate boundary edges whose adjacent facet
    is still in empty-ball configuration at the larger radius.
    """
    front_edges = {}
    front_queue = deque()

    for key, edge in list(boundary_edges.items()):
        center = _ball_center_from_reference(
            points[edge.i],
            points[edge.j],
            points[edge.opposite],
            rho,
            edge.ball_center,
        )
        if center is None:
            continue

        if not _ball_is_empty(center, rho, points, kdtree, (edge.i, edge.j, edge.opposite)):
            continue

        boundary_edges.pop(key)
        reactivated = Edge(edge.i, edge.j, edge.opposite, center)
        front_edges[key] = reactivated
        front_queue.append(reactivated)

    return front_queue, front_edges


def BallPivotingMethod(
    pcd: o3d.geometry.PointCloud,
    radii: o3d.utility.DoubleVector,
    **kwargs,
) -> o3d.geometry.TriangleMesh:
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)

    if len(points) < 3:
        return mesh

    if len(normals) != len(points):
        pcd = o3d.geometry.PointCloud(pcd)
        if not pcd.has_normals():
            pcd.estimate_normals()
        normals = np.asarray(pcd.normals)

    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normal_norms[normal_norms == 0.0] = 1.0
    normals = normals / normal_norms

    kdtree = KDTree(points)
    vertex_used = np.zeros(len(points), dtype=bool)
    open_edge_count = np.zeros(len(points), dtype=np.int32)

    triangles = []
    seen_triangles = set()
    used_dir_edges = set()
    boundary_edges = {}

    radii = [float(radius) for radius in radii]
    if not radii:
        return mesh

    first_radius = radii[0]
    while True:
        seed = _find_seed_triangle(points, normals, kdtree, first_radius, vertex_used)
        if seed is None:
            break

        seed_tri, seed_center = seed
        component_triangles = _grow_from_seed(
            points,
            normals,
            kdtree,
            first_radius,
            seed_tri,
            seed_center,
            vertex_used,
            open_edge_count,
            boundary_edges,
            triangles,
            seen_triangles,
            used_dir_edges,
        )
        print(f"  rho={first_radius}: component added {component_triangles} triangles (total: {len(triangles)})")

    for rho in radii[1:]:
        front_queue, front_edges = _reactivate_boundary_edges(points, kdtree, rho, boundary_edges)
        if not front_queue:
            print(f"  rho={rho}: no boundary edges reactivated")
            continue

        added = _expand_front(
            points,
            normals,
            kdtree,
            rho,
            front_queue,
            front_edges,
            boundary_edges,
            vertex_used,
            open_edge_count,
            triangles,
            seen_triangles,
            used_dir_edges,
        )
        print(f"  rho={rho}: reactivated pass added {added} triangles (total: {len(triangles)})")

    if triangles:
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    return mesh


if __name__ == "__main__":
    mesh_data = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    o3d.utility.random.seed(0)
    pcd = mesh_data.sample_points_poisson_disk(500)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(10)

    nn = np.asarray(pcd.compute_nearest_neighbor_distance())
    median_nn = float(np.median(nn))
    radii = o3d.utility.DoubleVector([median_nn * 1.5, median_nn * 3.0, median_nn * 6.0])
    print(
        f"Points: {len(pcd.points)}, NN median: {median_nn:.6f}, "
        f"radii: {[f'{radius:.6f}' for radius in radii]}"
    )

    generated_mesh = BallPivotingMethod(pcd, radii)
    generated_mesh.compute_vertex_normals()
    print(f"Generated {len(generated_mesh.triangles)} triangles")

    o3d.visualization.draw_geometries([generated_mesh, pcd], window_name="Ball Pivoting Result")

    if len(generated_mesh.triangles) > 0:
        ShowComparison(
            pcd,
            generated_mesh,
            BuiltinSurfaceReconstructionMethod.BALL_PIVOTING,
            radii=radii,
        )
