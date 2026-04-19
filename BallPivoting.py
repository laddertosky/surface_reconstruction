"""
Ball Pivoting Algorithm (BPA)-Surface Reconstruction
Based on:
    Bernardini et al. (1999), "The Ball-Pivoting Algorithm for Surface Reconstruction"
        http://mesh.brown.edu/taubin/pdfs/bernardini-etal-tvcg99.pdf
    Digne (2014), "An Analysis and Implementation of a Parallel Ball Pivoting Algorithm"
        https://www.ipol.im/pub/art/2014/81/article_lr.pdf

Done:
    Setup & data structures-Edge, KDTree, point state tracking
    _ball_center-computes where a ball of radius rho rests on the 3 points
    _find_seed_triangle-finds an initial triangle to start growing from
    _pivot-rolls the ball around an edge to find the next triangle
    Front expansion loop (runs until front is empty)
    Multi-radius passes (iterates over radii, resets front between passes)
Claude was used to help in debugging my code as well as the template of the code.

Note: seed search is O(N * M^2), so very large clouds (>~20k points) get a size-guard skip.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from Test import BuiltinSurfaceReconstructionMethod, BunnyPCD, ShowComparison

MAX_POINTS = 20000
TIME_BUDGET_SEC = 30.0  # give up and return partial mesh after this


@dataclass
class Edge:
    i: int
    j: int
    opposite: int  # vertex across from this edge
    ball_center: np.ndarray


def _ball_center(pi, pj, pk, rho, normals_ijk):
    """Ball center of radius rho on three points, or None."""
    # these ar e edge vectors from pi
    v0 = pj - pi
    v1 = pk - pi

    # just triangle normals
    normal = np.cross(v0, v1)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return None
    normal = normal / norm_len

    # circumcenter via barycentric coords
    d00 = np.dot(v0, v0)
    d11 = np.dot(v1, v1)
    d01 = np.dot(v0, v1)

    denom = 2.0 * (d00 * d11 - d01 * d01)
    if abs(denom) < 1e-12:
        return None

    s = d11 * (d00 - d01) / denom
    t = d00 * (d11 - d01) / denom

    circumcenter = pi + s * v0 + t * v1

    r_circ = np.linalg.norm(circumcenter - pi)
    gap = rho * rho - r_circ * r_circ
    if gap < 0:
        return None

    h = np.sqrt(gap)  # lift height

    # paper: orient n so each vertex normal has nonneg dot product
    dots = normals_ijk @ normal
    if np.all(dots >= 0):
        pass
    elif np.all(dots <= 0):
        normal = -normal
    else:
        return None  # vertex normals straddle triangle plane

    return circumcenter + h * normal


def _find_seed_triangle(points, normals, kdtree, rho, point_state):
    """Find first valid seed triangle among unused points."""
    n = len(points)
    for i in range(n):
        if point_state[i] != "unused":
            continue

        neighbors = kdtree.query_ball_point(points[i], 2.0 * rho)

        for ji, j in enumerate(neighbors):
            if j <= i:
                continue
            if point_state[j] == "inner":
                continue

            for k in neighbors[ji + 1:]:
                if k <= j:
                    continue
                if point_state[k] == "inner":
                    continue

                pi, pj, pk = points[i], points[j], points[k]
                normals_ijk = normals[np.array([i, j, k])]

                center = _ball_center(pi, pj, pk, rho, normals_ijk)
                if center is None:
                    continue

                #check no other points inside of ball
                nearby = kdtree.query_ball_point(center, rho - 1e-10)
                inside = [idx for idx in nearby if idx != i and idx != j and idx != k]
                if inside:
                    continue

                # per-vertex compatibility (paper sec 2.3): tri normal must dot > 0 with each vertex normal
                tri_normal = np.cross(pj - pi, pk - pi)
                tri_normal_len = np.linalg.norm(tri_normal)
                if tri_normal_len < 1e-12:
                    continue
                tri_normal = tri_normal / tri_normal_len

                dots = normals_ijk @ tri_normal
                if np.all(dots > 0):
                    return (i, j, k), center
                if np.all(dots < 0):
                    return (i, k, j), center  # flipped winding also compatible
                # else: incompatible, try next k

    return None


def _pivot(points, normals, kdtree, rho, edge, point_state):
    """Roll ball around edge, return (k, center) or None."""
    pi = points[edge.i]
    pj = points[edge.j]
    midpoint = 0.5 * (pi + pj)

    edge_vec = pj - pi
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        return None
    edge_axis = edge_vec / edge_len

    # project old center onto plane perp to edge
    old_rel = edge.ball_center - midpoint
    old_proj = old_rel - np.dot(old_rel, edge_axis) * edge_axis
    old_proj_len = np.linalg.norm(old_proj)
    if old_proj_len < 1e-12:
        return None
    old_proj_unit = old_proj / old_proj_len

    perp = np.cross(edge_axis, old_proj_unit)  # 2nd basis vec

    neighbors = kdtree.query_ball_point(midpoint, 2.0 * rho)

    candidates = []

    for k in neighbors:
        if k == edge.i or k == edge.j or k == edge.opposite:
            continue
        if point_state[k] == "inner":
            continue

        normals_ijk = normals[np.array([edge.i, edge.j, k])]
        center = _ball_center(pi, pj, points[k], rho, normals_ijk)
        if center is None:
            continue

        # per-vertex compatibility: output tri is (j, i, k) with normal = -tri_normal.
        # need -tri_normal dot > 0 with each vertex normal, i.e. tri_normal dot < 0 with each
        tri_normal = np.cross(pj - pi, points[k] - pi)
        tri_normal_len = np.linalg.norm(tri_normal)
        if tri_normal_len < 1e-12:
            continue
        tri_normal = tri_normal / tri_normal_len
        dots = normals_ijk @ tri_normal
        if np.any(dots >= 0):
            continue

        # pivot angle
        new_rel = center - midpoint
        new_proj = new_rel - np.dot(new_rel, edge_axis) * edge_axis
        new_proj_len = np.linalg.norm(new_proj)
        if new_proj_len < 1e-12:
            continue
        new_proj_unit = new_proj / new_proj_len

        cos_angle = np.clip(np.dot(old_proj_unit, new_proj_unit), -1.0, 1.0)
        sin_angle = np.dot(perp, new_proj_unit)
        angle = np.arctan2(sin_angle, cos_angle)

        if angle <= 1e-10:
            angle += 2.0 * np.pi

        candidates.append((angle, k, center))

    candidates.sort(key=lambda x: x[0])

    for angle, k, center in candidates:
        nearby = kdtree.query_ball_point(center, rho - 1e-10)
        inside = [idx for idx in nearby if idx != edge.i and idx != edge.j and idx != k]
        if not inside:
            return k, center

    return None


def _add_triangle(tri, triangles, seen_triangles, used_dir_edges):
    """Try to add a triangle (a, b, c); returns True if added. Rejects if any directed edge is already used (would violate manifold)."""
    a, b, c = tri
    key = tuple(sorted((a, b, c)))
    if key in seen_triangles:
        return False
    dir_edges = [(a, b), (b, c), (c, a)]
    if any(e in used_dir_edges for e in dir_edges):
        return False
    seen_triangles.add(key)
    used_dir_edges.update(dir_edges)
    triangles.append(tri)
    return True


def _bpa_single_radius(points, normals, kdtree, rho, point_state, triangles, seen_triangles, used_dir_edges, deadline):
    """Run BPA for one ball radius, modifies state in place."""
    tri_cap = 10 * len(points)  # safety cap, real meshes are ~2x V
    stall_limit = 10  # bail if this many seeds produce zero new triangles in a row
    stall_count = 0
    while True:
        if time.time() > deadline:
            print(f"  rho={rho}: time budget reached, bailing")
            return False
        if len(triangles) > tri_cap:
            print(f"  rho={rho}: hit triangle cap ({tri_cap}), bailing")
            break
        if stall_count >= stall_limit:
            break
        seed = _find_seed_triangle(points, normals, kdtree, rho, point_state)
        if seed is None:
            break

        (i, j, k), ball_center = seed
        if not _add_triangle((i, j, k), triangles, seen_triangles, used_dir_edges):
            # reject overlapping/duplicate seed, mark points and move on
            point_state[i] = "front"
            point_state[j] = "front"
            point_state[k] = "front"
            continue
        point_state[i] = "front"
        point_state[j] = "front"
        point_state[k] = "front"

        front = deque()
        edge_map = {}
        front_edge_count = {}

        for e in [Edge(i, j, k, ball_center), Edge(j, k, i, ball_center), Edge(k, i, j, ball_center)]:
            front.append(e)
            edge_map[(e.i, e.j)] = e
            front_edge_count[e.i] = front_edge_count.get(e.i, 0) + 1
            front_edge_count[e.j] = front_edge_count.get(e.j, 0) + 1

        tri_count_start = len(triangles)
        while front:
            if len(triangles) > tri_cap or time.time() > deadline:
                break
            edge = front.popleft()
            if (edge.i, edge.j) not in edge_map:
                continue

            result = _pivot(points, normals, kdtree, rho, edge, point_state)

            if result is None:  # boundary edge
                del edge_map[(edge.i, edge.j)]
                front_edge_count[edge.i] -= 1
                front_edge_count[edge.j] -= 1
                continue

            new_k, new_center = result
            if not _add_triangle((edge.j, edge.i, new_k), triangles, seen_triangles, used_dir_edges):
                # overlaps existing mesh; treat edge as boundary
                del edge_map[(edge.i, edge.j)]
                front_edge_count[edge.i] -= 1
                front_edge_count[edge.j] -= 1
                continue
            del edge_map[(edge.i, edge.j)]
            front_edge_count[edge.i] -= 1
            front_edge_count[edge.j] -= 1

            if point_state[new_k] == "unused":
                point_state[new_k] = "front"
                front_edge_count[new_k] = front_edge_count.get(new_k, 0)

            new_edges = [
                Edge(edge.i, new_k, edge.j, new_center),
                Edge(new_k, edge.j, edge.i, new_center),
            ]
            for ne in new_edges:
                reverse = (ne.j, ne.i)
                if reverse in edge_map:
                    del edge_map[reverse]
                    front_edge_count[ne.i] -= 1
                    front_edge_count[ne.j] -= 1
                else:
                    front.append(ne)
                    edge_map[(ne.i, ne.j)] = ne
                    front_edge_count[ne.i] = front_edge_count.get(ne.i, 0) + 1
                    front_edge_count[ne.j] = front_edge_count.get(ne.j, 0) + 1

            for p in [edge.i, edge.j, new_k]:
                if front_edge_count.get(p, 0) <= 0 and point_state[p] == "front":
                    point_state[p] = "inner"

        added = len(triangles) - tri_count_start
        stall_count = stall_count + 1 if added == 0 else 0
        print(f"  rho={rho}: component added {added} triangles (total: {len(triangles)})")
    return True


def BallPivotingMethod(pcd: o3d.geometry.PointCloud, radii: o3d.utility.DoubleVector, **kwargs) -> o3d.geometry.TriangleMesh:
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    if len(points) > MAX_POINTS:
        print(f"BallPivoting: skipping {len(points)} points (> {MAX_POINTS} cap)")
        return o3d.geometry.TriangleMesh()


    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    kdtree = KDTree(points)
    point_state = ["unused"] * len(points)
    triangles = []
    seen_triangles = set()
    used_dir_edges = set()
    deadline = time.time() + TIME_BUDGET_SEC

    for rho in radii:
        # reset front points for next radius pass
        for idx in range(len(point_state)):
            if point_state[idx] == "front":
                point_state[idx] = "unused"

        finished = _bpa_single_radius(points, normals, kdtree, float(rho), point_state, triangles, seen_triangles, used_dir_edges, deadline)
        if not finished:
            break

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    if triangles:
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    return mesh


if __name__ == "__main__":
    mesh_data = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
    o3d.utility.random.seed(0)
    pcd = mesh_data.sample_points_poisson_disk(500)
    pcd.estimate_normals()

    nn = np.asarray(pcd.compute_nearest_neighbor_distance())
    median_nn = float(np.median(nn))
    radii = o3d.utility.DoubleVector([median_nn * 1.5, median_nn * 3, median_nn * 6])
    print(f"Points: {len(pcd.points)}, NN median: {median_nn:.6f}, radii: {[f'{r:.6f}' for r in radii]}")

    generated_mesh = BallPivotingMethod(pcd, radii)
    generated_mesh.compute_vertex_normals()
    print(f"Generated {len(generated_mesh.triangles)} triangles")

    o3d.visualization.draw_geometries([generated_mesh, pcd], window_name="Ball Pivoting Result")

    if len(generated_mesh.triangles) > 0:
        ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.BALL_PIVOTING, radii=radii)
