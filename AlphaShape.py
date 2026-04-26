"""
Reference:
Edelsbrunner, H., Mücke, E. (1994)
Three-dimensional alpha shapes.
https://arxiv.org/abs/math/9410208

Used for: surface reconstruction / alpha shape algorithm
"""

import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay 
from scipy.spatial import cKDTree

from Test import BuiltinSurfaceReconstructionMethod, BunnyPCD, ShowComparison

def sort_face(a, b, c):
    return tuple(sorted([int(a), int(b), int(c)]))

def oriented_face_from_tetra(v, i, j, k, opp):
    """
    Orient triangle (i, j, k) so its normal points away from tetrahedron
    (i, j, k, opp). If the current winding points toward the opposite vertex opp,
    flip the triangle.

    Parameters
    ----------
    v : (N, 3) array
        Array of point coordinates.
    i, j, k : int
        Vertex indices of the face.
    opp : int
        Vertex index opposite the face in the tetrahedron.

    Returns
    -------
    tuple of int
        The face vertex indices in outward-facing order.
    """
    pi = v[i]
    pj = v[j]
    pk = v[k]
    po = v[opp]

    normal = np.cross(pj - pi, pk - pi)

    if np.dot(normal, po - pi) > 0:
        return (i, k, j)
    
    return (i, j, k)

def prepare_delaunay_data(pcd):
    """
    Build the 3D Delaunay tetrahedralization and its connectivity structure
    from a point cloud.

    Parameters
    ------------
    pcd: o3d.geometry.PointCloud

    Returns
    ------------
    data: dict
        - 'vertices'               : (N, 3) float array
        - 'tetras'                 : (M, 4) int array of tetrahedra (vertex indices)
        - 'triangles'              : (F, 3) int array of unique triangle faces (vertex indices)
        - 'tetra_to_faces'         : list of lists of face IDs per tetra
        - 'tetra_to_oriented_faces': list of lists of oriented face tuples per tetra
        - 'face_to_tetras'         : dict mapping face ID -> list of tetra IDs
    """

    # 3d Delaunay tetrahedralization
    vertices = np.asarray(pcd.points)
    tri = Delaunay(vertices)
    tetras = tri.simplices

    face_to_id = {}
    triangles = []
    tetra_to_faces = []
    tetra_to_oriented_faces = []
    face_to_tetras = {}

    # Process every tetrahedron
    for tid, tet in enumerate(tetras):
        # Vertices of the tetrahedron
        a, b, c, d = map(int, tet)

        # Fenerate 4 triangular faces of tetra
        tet_faces = [
            sort_face(a, b, c),
            sort_face(a, b, d),
            sort_face(a, c, d),
            sort_face(b, c, d),
        ]

        tet_oriented_faces = [
            oriented_face_from_tetra(vertices, a, b, c, d),
            oriented_face_from_tetra(vertices, a, b, d, c),
            oriented_face_from_tetra(vertices, a, c, d, b),
            oriented_face_from_tetra(vertices, b, c, d, a),
        ]

        tet_face_ids = []
        for f in tet_faces:
            # Add faces to the global face list
            if f not in face_to_id:
                face_to_id[f] = len(triangles)
                triangles.append(f)

            # Record which tetra use each face
            fid = face_to_id[f]
            tet_face_ids.append(fid)

            face_to_tetras.setdefault(fid, []).append(tid)
        
        # Record the 4 faces of the tetra
        tetra_to_faces.append(tet_face_ids)
        tetra_to_oriented_faces.append(tet_oriented_faces)

    data = {
        "vertices": vertices,
        "triangles": np.array(triangles),
        "tetras": tetras,
        "face_to_tetras": dict(face_to_tetras),
        "tetra_to_faces": tetra_to_faces,
        "tetra_to_oriented_faces": tetra_to_oriented_faces,
    }      
            
    return data

def tetra_circumsphere(tetra_vs):
    """
    Compute the circumsphere of a tetrahedron.

    Parameters
    ----------
    tetra_vs : array-like of shape (4, 3)
        The four 3D vertices of the tetrahedron.

    Returns
    -------
    center : ndarray of shape (3,) or None
        Circumsphere center, or None if the tetrahedron is degenerate.
    radius : float
        Circumsphere radius, or np.inf if degenerate.
    valid : bool
        Whether the circumsphere was computed successfully.
    """

    a, b, c, d = [np.asarray(v) for v in tetra_vs]

    A = np.vstack([
        b - a,
        c - a,
        d - a
    ])

    detA = np.linalg.det(A)
    if abs(detA) < 1e-12:
        return None, np.inf, False
    
    rhs = 0.5 * np.array([
        np.dot(b, b) - np.dot(a, a),
        np.dot(c, c) - np.dot(a, a),
        np.dot(d, d) - np.dot(a, a),
    ])

    try:
        center = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None, np.inf, False
    
    radius = np.linalg.norm(center - a)

    return center, radius, True

def triangle_circumsphere_radius(tri_v):
    """
    Compute triangles cirmcumsphere, circumradius, and unit normal of a 3D triangle.

    Parameters
    ----------
    tri_v: (3, 3) array

    Returns
    -------
    center : (3,) ndarray or None
    radius : float
    normal : (3,) ndarray or None
    valid : bool
    """

    a, b, c = [np.asarray(v) for v in tri_v]

    ab = b - a
    ac = c - a
    
    cross = np.cross(b - a, c - a)
    norm_cross = np.linalg.norm(cross)

    if norm_cross < 1e-12:
        return None, np.inf, None, False
    
    normal = cross / norm_cross

    a_b = np.dot(ab, ab)
    a_c = np.dot(ac, ac)

    center = a + (np.cross(cross, ab) * a_c + np.cross(ac, cross) * a_b) / (2.0 * norm_cross**2)

    radius = np.linalg.norm(center - a)

    return center, radius, normal, True

def compute_alpha_exposed(data, alpha):
    """
    Classify Delaunay faces using face circumradius + incident tetra circumradii.

    Returns
    -------
    regular_faces : list[int]
    singular_faces : list[int]
    """

    vertices = data['vertices']
    regular_faces = []
    singular_faces = []

    tetra_radii = []
    for tet in data['tetras']:
        tet_v = vertices[tet]
        _, r, valid = tetra_circumsphere(tet_v)
        tetra_radii.append(r if valid else np.inf)
    tetra_radii = np.asarray(tetra_radii)

    for fid, tri in enumerate(data['triangles']):
        tri_v = vertices[list(map(int, tri))]

        _, r_face, _, valid = triangle_circumsphere_radius(tri_v)
        if not valid or r_face >= alpha:
            continue

        incident_tids = data['face_to_tetras'][fid]
        n_incidents = len(incident_tids)

        n_interior = sum(
            tetra_radii[tid] < alpha 
            for tid in incident_tids
        )

        if n_incidents == 1:
            regular_faces.append(fid) 
        elif n_incidents == 2:
            if n_interior == 1:
                regular_faces.append(fid)
            elif n_interior == 0 and r_face < alpha * 0.5:
                singular_faces.append(fid)

    return regular_faces, singular_faces, tetra_radii

def extract_surface(data, face_ids, alpha, tetra_radii, mesh_centroid = None):
    """
    Extract oriented boundary triangles from selected face IDs.

    Parameters
    ----------
    data : dict
        Output of prepare_delaunay_data().
    face_ids : list of int
        Face IDs selected by compute_alpha_exposed (regular + singular).
    alpha : float
        Alpha value used to identify the interior tetrahedron per face.
    tetra_radii : ndarray
        Precomputed circumradii per tetrahedron, from compute_alpha_exposed.
    mesh_centroid : (3,) array, optional
        Used to orient singular faces. Defaults to vertex mean.

    Returns
    -------
    surface : (K, 3) int array
        Oriented boundary triangles.
    """

    vertices = data["vertices"]

    if mesh_centroid is None:
        mesh_centroid = vertices.mean(axis=0)

    surface = []

    for fid in face_ids:
        incident_tetras = data["face_to_tetras"][fid]

        chosen_tid = None
        for tid in incident_tetras:
            if tetra_radii[tid] < alpha:
                chosen_tid = tid
                break

        if chosen_tid is None:
            i, j, k = map(int, data['triangles'][fid])
            pi, pj, pk = vertices[i], vertices[j], vertices[k]
            normal = np.cross(pj - pi, pk - pi)
            face_center = (pi + pj + pk) / 3.0

            if np.dot(normal, face_center - mesh_centroid) < 0:
                oriented_face = (i, k, j)
            else:
                oriented_face = (i, j, k)
        else:
            local_face_ids = data["tetra_to_faces"][chosen_tid]
            local_idx = local_face_ids.index(fid)
            oriented_face = data["tetra_to_oriented_faces"][chosen_tid][local_idx]


        surface.append(oriented_face)

    if len(surface) == 0:
        return np.empty((0, 3), dtype=int)

    return np.asarray(surface, dtype=int)

def normalize_pcd(pcd):
    """
    Scale point cloud to fit inside unit sphere.
    """
    pts = np.asarray(pcd.points, dtype=float)
    center = pts.mean(axis=0)
    pts = pts - center

    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
   
    normalized = o3d.geometry.PointCloud()
    normalized.points = o3d.utility.Vector3dVector(pts)

    if pcd.has_normals():
        normalized.normals = pcd.normals

    return normalized, center, scale

def AlphaShapeMethod(pcd: o3d.geometry.PointCloud, alpha: float, **kwargs) -> o3d.geometry.TriangleMesh:

    # Normalize pcd
    pcd_norm, center, scale = normalize_pcd(pcd)

    # Construct Delaunay complex
    data = prepare_delaunay_data(pcd_norm)
    regular_ids, singular_ids, tetra_radii = compute_alpha_exposed(data, alpha)

    centroid = np.asarray(pcd_norm.points).mean(axis=0)
    surface = extract_surface(data, regular_ids + singular_ids, alpha, tetra_radii, mesh_centroid=centroid)

    # Build mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(data['vertices'])
    mesh.triangles = o3d.utility.Vector3iVector(surface)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.orient_triangles()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    
    mesh.scale(scale, center=(0.0, 0.0, 0.0))
    mesh.translate(center)

    return mesh

# Helper function to help to estimate an alpha
def estimate_alpha(pcd, multiplier=3.0):

    pts = np.asarray(pcd.points, dtype=float)
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    median_nn_dist = np.median(dists[:, 1])

    return median_nn_dist * multiplier

if __name__ == "__main__":
    pcd = BunnyPCD()
    generated_mesh = AlphaShapeMethod(pcd, alpha=0.5)
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.ALPHA_SHAPE, alpha=0.5)
