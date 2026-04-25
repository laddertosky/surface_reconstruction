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

def alpha_sphere_centers(tri_v, alpha):
    """
    Return the candidates centers of radius-alpha spheres through the triangles
    """

    center, r_face, normal, valid = triangle_circumsphere_radius(tri_v)

    if not valid or alpha + 1e-12 < r_face:
        return [], r_face, False
    
    delta = max(alpha**2 - r_face**2, 0.0)
    h = np.sqrt(delta)

    if h < 1e-12:
        return [center], r_face, True
    
    c1 = center + h * normal
    c2 = center - h * normal

    return [c1, c2], r_face, True

def is_empty_alpha_sphere(kdtree, center, alpha, face_vertex_ids):
    """
    Check the sphere centered at `center` with radius `alpha`
    is empty of all points except the face's own vertices.
    """

    indices = kdtree.query_ball_point(center, alpha - 1e-12)
    face_set = set(map(int, face_vertex_ids))

    return all(i in face_set for i in indices)

def compute_alpha_exposed(data, alpha):
    """
    Test all Delaunay triangles for alpha-exposure
    """

    vertices = data['vertices']
    tree = cKDTree(vertices)
    regular_faces = []
    singular_faces = []

    for fid, tri in enumerate(data['triangles']):
        tri_ids = tuple(map(int, tri))
        tri_v = vertices[list(tri_ids)]

        centers, r_face, valid = alpha_sphere_centers(tri_v, alpha)
        if not valid:
            continue

        n_incident_tetras = len(data['face_to_tetras'][fid])

        empty_sides = [
            is_empty_alpha_sphere(tree, c, alpha, tri_ids)
            for c in centers
        ]

        n_empty = sum(empty_sides)

        if n_incident_tetras == 1:
            if n_empty >= 1:
                regular_faces.append(fid)
        else:
            if n_empty == 1:
                regular_faces.append(fid)
            elif n_empty == 2:
                singular_faces.append(fid)

    return regular_faces, singular_faces

def extract_surface(data, face_ids, alpha):
    """
    Extract oriented boundary triangles from selected face IDs.

    Parameters
    ----------
    data : dict
        Output of prepare_delaunay_data().
    face_ids : list of int
        Face IDs to include in the surface (regular or singular).
    alpha : float
        Alpha value used to identify the interior tetrahedron per face.

    Returns
    -------
    surface : (K, 3) int array
        Oriented boundary triangles.
    """

    vertices = data["vertices"]
    surface = []

    for fid in face_ids:
        incident_tetras = data["face_to_tetras"][fid]

        chosen_tid = None
        for tid in incident_tetras:
            tet_v = vertices[data["tetras"][tid]]
            _, r, valid = tetra_circumsphere(tet_v)
            if valid and r < alpha:
                chosen_tid = tid
                break

        if chosen_tid is None:
            chosen_tid = incident_tetras[0]

        local_face_ids = data["tetra_to_faces"][chosen_tid]
        local_idx = local_face_ids.index(fid)

        oriented_face = data["tetra_to_oriented_faces"][chosen_tid][local_idx]
        surface.append(oriented_face)

    if len(surface) == 0:
        return np.empty((0, 3), dtype=int)

    return np.asarray(surface, dtype=int)

def AlphaShapeMethod(pcd: o3d.geometry.PointCloud, alpha: float, **kwargs) -> o3d.geometry.TriangleMesh:

    # Construct Delaunay complex
    data = prepare_delaunay_data(pcd)

    regular_ids, singular_ids = compute_alpha_exposed(data, alpha)
    surface = extract_surface(data, regular_ids, alpha)

    # Build mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(data['vertices'])
    mesh.triangles = o3d.utility.Vector3iVector(surface)

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.orient_triangles()
    mesh.remove_non_manifold_edges() 
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    print(mesh.is_orientable())
    print(mesh.is_watertight())
    
    return mesh

if __name__ == "__main__":
    pcd = BunnyPCD()
    generated_mesh = AlphaShapeMethod(pcd, alpha=0.5)
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.ALPHA_SHAPE, alpha=0.5)
