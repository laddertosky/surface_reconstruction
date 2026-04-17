"""
Ball Pivoting Algorithm (BPA)-Surface Reconstruction
Based on:
    Bernardini et al. (1999), "The Ball-Pivoting Algorithm for Surface Reconstruction"
        http://mesh.brown.edu/taubin/pdfs/bernardini-etal-tvcg99.pdf
    Digne (2014), "An Analysis and Implementation of a Parallel Ball Pivoting Algorithm"
        https://www.ipol.im/pub/art/2014/81/article_lr.pdf

Status: This is currently a work in progres implementation

Done:
    Setup & data structures-Edge, KDTree, point state tracking
    _ball_center-computes where a ball of radius rho rests on the 3 points
    _find_seed_triangle-finds an initial triangle to start growing from
    _pivot-rolls the ball around an edge to find the next triangle
Claude was used to help in debugging my code as well as the template of the code.
TODO:
    Full front expansion loop (remove the 10-pivot cap, run until front is empty)
    Multi-radius passes (iterate over radii, reset boundary points between passes)
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

from Test import BuiltinSurfaceReconstructionMethod, BunnyPCD, ShowComparison


@dataclass
class Edge:
    i: int #1st vertex index
    j: int #2nd vertex index
    opposite: int #3rd vertex of the triangle that created this edge
    ball_center: np.ndarray #(3,) center of the ball when this edge was formed


def _ball_center(pi, pj, pk, rho, normals_ijk):
    """
    Compute the center of a ball of radius rho resting on three points.
    Returns the ball center on the side consistent with the average normal, or None if the circumradius exceeds rho (since ball can't touch all three)
    """
    # these ar e edge vectors from pi
    v0 = pj - pi
    v1 = pk - pi

    # just triangle normals
    normal = np.cross(v0, v1)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return None
    normal = normal / norm_len

    # Circumcenter in the plane of the triangle with formula circumcenter = pi + (|v1|^2 (v0 x v1) x v0 + |v0|^2 v1 x (v0 x v1)) / (2|v0 x v1|^2)
    d00 = np.dot(v0, v0)
    d11 = np.dot(v1, v1)
    d01 = np.dot(v0, v1)

    denom = 2.0 * (d00 * d11 - d01 * d01)
    if abs(denom) < 1e-12:
        return None

    s = (d00 * d11 - d11 * d01) / denom
    t = (d11 * d00 - d00 * d01) / denom

    circumcenter = pi + s * v0 + t * v1

    # circumradius
    r_circ = np.linalg.norm(circumcenter - pi)

    # the check for if ball is large enough since
    gap = rho * rho - r_circ * r_circ
    if gap < 0:
        return None

    #perpendicular needs to be lifted to the triangle plane
    h = np.sqrt(gap)

    #compute average normal of 3 vertices to find which side
    avg_normal = normals_ijk.mean(axis=0)
    avg_normal_len = np.linalg.norm(avg_normal)
    if avg_normal_len < 1e-12:
        avg_normal = normal
    else:
        avg_normal = avg_normal / avg_normal_len

    # choose side that align with average vertex normal
    if np.dot(normal, avg_normal) >= 0:
        center = circumcenter + h * normal
    else:
        center = circumcenter - h * normal

    return center


def _find_seed_triangle(points, normals, kdtree, rho, point_state):
    """
    Find three unused/front points that a ball of radius rho can rest on with no other points inside the ball.
    Returns ((i, j, k), ball_center) or None.
    """
    n = len(points)
    for i in range(n):
        if point_state[i] == "inner":
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

                #consistent winding meaning that triangle normal should align with avg vertex normal
                tri_normal = np.cross(pj - pi, pk - pi)
                tri_normal_len = np.linalg.norm(tri_normal)
                if tri_normal_len < 1e-12:
                    continue
                tri_normal = tri_normal / tri_normal_len

                avg_normal = normals_ijk.mean(axis=0)
                if np.dot(tri_normal, avg_normal) < 0:
                    # Swap j and k to flip winding
                    return (i, k, j), center

                return (i, j, k), center

    return None


def _pivot(points, normals, kdtree, rho, edge, point_state):
    """
    Pivot ball around the given edge to find the next point.
    Returns (k, new_ball_center) for the best candidate, or None if no valid pivot exists (the edge is a boundary).
    """
    pi = points[edge.i]
    pj = points[edge.j]
    midpoint = 0.5 * (pi + pj)

    # edge axis
    edge_vec = pj - pi
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        return None
    edge_axis = edge_vec / edge_len

    # project the old ball center into plane perpendicular to edge at the midpoint
    old_rel = edge.ball_center - midpoint
    old_proj = old_rel - np.dot(old_rel, edge_axis) * edge_axis
    old_proj_len = np.linalg.norm(old_proj)
    if old_proj_len < 1e-12:
        return None
    old_proj_unit = old_proj / old_proj_len

    # the second basis vector for the angle plane (basically the right-hand rule)
    perp = np.cross(edge_axis, old_proj_unit)

    neighbors = kdtree.query_ball_point(midpoint, 2.0 * rho)

    best_angle = float('inf')
    best_k = None
    best_center = None

    for k in neighbors:
        if k == edge.i or k == edge.j or k == edge.opposite:
            continue
        if point_state[k] == "inner":
            continue

        normals_ijk = normals[np.array([edge.i, edge.j, k])]
        center = _ball_center(pi, pj, points[k], rho, normals_ijk)
        if center is None:
            continue

        # empty ball check
        nearby = kdtree.query_ball_point(center, rho - 1e-10)
        inside = [idx for idx in nearby if idx != edge.i and idx != edge.j and idx != k]
        if inside:
            continue

        # compute pivot anglewhich is the angle from the old ball center to new center as rotation around the edge axis
        new_rel = center - midpoint
        new_proj = new_rel - np.dot(new_rel, edge_axis) * edge_axis
        new_proj_len = np.linalg.norm(new_proj)
        if new_proj_len < 1e-12:
            continue
        new_proj_unit = new_proj / new_proj_len

        cos_angle = np.clip(np.dot(old_proj_unit, new_proj_unit), -1.0, 1.0)
        sin_angle = np.dot(perp, new_proj_unit)
        angle = np.arctan2(sin_angle, cos_angle)

        #we want the smallest positive angle (first point the ball touches)
        if angle <= 1e-10:
            angle += 2.0 * np.pi

        if angle < best_angle:
            best_angle = angle
            best_k = k
            best_center = center

    if best_k is None:
        return None

    return best_k, best_center


def BallPivotingMethod(pcd: o3d.geometry.PointCloud, radii: o3d.utility.DoubleVector, **kwargs) -> o3d.geometry.TriangleMesh:
    # TODO main loop + multi-radius not yet implemented
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # normalize normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    kdtree = KDTree(points)
    point_state = ["unused"] * len(points)
    triangles = []

    # placeholder-just find a seed and do a few pivots to test
    rho = float(radii[0])
    seed = _find_seed_triangle(points, normals, kdtree, rho, point_state)
    if seed is not None:
        (i, j, k), ball_center = seed
        triangles.append((i, j, k))
        point_state[i] = "front"
        point_state[j] = "front"
        point_state[k] = "front"

        front = deque()
        edge_map = {}
        for e in [Edge(i, j, k, ball_center), Edge(j, k, i, ball_center), Edge(k, i, j, ball_center)]:
            front.append(e)
            edge_map[(e.i, e.j)] = e

        #run couple of pivots jsut to test for now
        max_pivots = 10
        count = 0
        while front and count < max_pivots:
            edge = front.popleft()
            if (edge.i, edge.j) not in edge_map:
                continue

            result = _pivot(points, normals, kdtree, rho, edge, point_state)

            if result is None:
                del edge_map[(edge.i, edge.j)]
                continue

            new_k, new_center = result
            #new triangle and reverse the pivoted edge + new point
            triangles.append((edge.j, edge.i, new_k))
            del edge_map[(edge.i, edge.j)]

            if point_state[new_k] == "unused":
                point_state[new_k] = "front"

            #two new edges from the new triangle
            new_edges = [
                Edge(edge.i, new_k, edge.j, new_center),
                Edge(new_k, edge.j, edge.i, new_center),
            ]
            for ne in new_edges:
                reverse = (ne.j, ne.i)
                if reverse in edge_map:
                    del edge_map[reverse]
                else:
                    front.append(ne)
                    edge_map[(ne.i, ne.j)] = ne

            count += 1

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    if triangles:
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32))
    return mesh


if __name__ == "__main__":
    pcd = BunnyPCD()
    # the radii should be based on point spacing for 100-point bunny, NN median ~0.02
    radii = o3d.utility.DoubleVector([0.04, 0.08, 0.16])
    generated_mesh = BallPivotingMethod(pcd, radii)
    generated_mesh.compute_vertex_normals()
    print(f"Generated {len(generated_mesh.triangles)} triangles")

    o3d.visualization.draw_geometries([generated_mesh, pcd], window_name="Ball Pivoting Result")

    if len(generated_mesh.triangles) > 0:
        ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.BALL_PIVOTING, radii=radii)
