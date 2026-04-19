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

from Test import BuiltinSurfaceReconstructionMethod, BunnyPCD, ShowComparison


def sort_edge(a, b):
    return tuple(sorted([int(a), int(b)]))

def sort_face(a, b, c):
    return tuple(sorted([int(a), int(b), int(c)]))

def prepare_delaunay_data(pcd):
    """
    Build the 3D Delaunay tetrahedralization and its connectivity structure
    from a point cloud.

    Given a point cloud, this function computes the 3D Delaunay tetrahedralization
    using SciPy and then constructs a combinatorial structure on top of it:
    - input vertices and Delaunay tetrahedra
    - unique edges and triangular faces
    - adjacency maps between all entities:
        * tetra  -> faces, edges
        * face   -> tetras, edges
        * edge   -> faces
        * vertex -> tetras, faces, edges

    Parameters
    ------------
    pcd: o3d.geometry.PointCloud

    Returns
    ------------
    data: dict
        - 'vertices'          : (N, 3) float array
        - 'tetras'            : (M, 4) int array of tetrahedra (vertex indices)
        - 'tetra_neighbors'   : (M, 4) int array of neighboring tetra indices
        - 'edges'             : (E, 2) int array of unique edges (vertex indices)
        - 'triangles'         : (F, 3) int array of unique triangle faces (vertex indices)
        - 'tetra_to_edges'    : list of lists of edge IDs per tetra
        - 'tetra_to_faces'    : list of lists of face IDs per tetra
        - 'face_to_edges'     : list of lists of edge IDs per face
        - 'face_to_tetras'    : dict mapping face ID -> list of tetra IDs
        - 'edge_to_faces'     : dict mapping edge ID -> list of face IDs
        - 'vertex_to_edges'   : dict mapping vertex ID -> sorted list of edge IDs
        - 'vertex_to_faces'   : dict mapping vertex ID -> sorted list of face IDs
        - 'vertex_to_tetras'  : dict mapping vertex ID -> sorted list of tetra IDs
        - 'vertex_to_simplex' : (N,) int array; for each vertex, one incident tetra index
        - 'scipy_delaunay'    : the original SciPy Delaunay object
    """

    # 3d Delaunay tetrahedralization
    vertices = np.asarray(pcd.points)
    tri = Delaunay(vertices)
    tetras = tri.simplices
    tetras_neighbors = tri.neighbors

    # --- Global containers ---
    edge_to_id = {}
    face_to_id ={}
    edges = []
    triangles = []

    # --- Adjacency containers ---
    tetra_to_faces = []
    tetra_to_edges = []

    face_to_tetras = {}
    edge_to_faces = {}

    vertex_to_edges = {}
    vertex_to_faces = {}
    vertex_to_tetras = {}

    # Process every tetrahedron
    for tid, tet in enumerate(tetras):
        # Vertices of the tetrahedron
        a, b, c, d = map(int, tet)

        # Record which tetrahedra touch each vertex
        for v in tet:
            vertex_to_tetras.setdefault(v, set()).add(tid)
            
        # Generate 6 edges of a tetra
        tet_edges = [
            sort_edge(a, b),
            sort_edge(a, c),
            sort_edge(a, d),
            sort_edge(b, c),
            sort_edge(b, d),
            sort_edge(c, d),
        ]

        tet_edge_ids = []
        for e in tet_edges:
            # Add edge to the global edge list if not have seen before
            if e not in edge_to_id:
                edge_to_id[e] = len(edges)
                edges.append(e)

            # Record which edge belong to this tetra
            eid = edge_to_id[e]
            tet_edge_ids.append(eid)

            # Record which vertices touch which edges
            i, j = e
            vertex_to_edges.setdefault(i, set()).add(eid)
            vertex_to_edges.setdefault(j, set()).add(eid)
        
        # Record the 6 edges od this tetra
        tetra_to_edges.append(tet_edge_ids)

        # Fenerate 4 triangular faces of tetra
        tet_faces = [
            sort_face(a, b, c),
            sort_face(a, b, d),
            sort_face(a, c, d),
            sort_face(b, c, d),
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

            # Record which vertices touch which faces
            i, j, k = f
            vertex_to_faces.setdefault(i, set()).add(fid)
            vertex_to_faces.setdefault(j, set()).add(fid)
            vertex_to_faces.setdefault(k, set()).add(fid)
        
        # Record the 4 faces of the tetra
        tetra_to_faces.append(tet_face_ids)

    # Build face to edge links
    face_to_edges = []
    for fid, f in enumerate(triangles):
        i, j, k = f
        f_edges = [
            edge_to_id[sort_edge(i, j)],
            edge_to_id[sort_edge(i, k)],
            edge_to_id[sort_edge(j, k)],
        ]
        face_to_edges.append(f_edges)

        # Record face to edge adjacency
        for eid in f_edges:
            edge_to_faces.setdefault(eid, []).append(fid)
            
    # Convert sets to sorted list
    vertex_to_edges = {v: sorted(ids) for v, ids in vertex_to_edges.items()}
    vertex_to_faces = {v: sorted(ids) for v, ids in vertex_to_faces.items()}
    vertex_to_tetras = {v: sorted(ids) for v, ids in vertex_to_tetras.items()}

    data = {
        "vertices": vertices,
        "tetras": tetras,
        "tetras_neighbors": tetras_neighbors,
        "edges": np.array(edges),
        "triangles": np.array(triangles),

        "tetra_to_edges": tetra_to_edges,
        "tetra_to_faces": tetra_to_faces,
        
        "face_to_edges": face_to_edges,
        "face_to_tetras": dict(face_to_tetras),

        "edge_to_faces": dict(edge_to_faces),

        "vertex_to_edges": vertex_to_edges,
        "vertex_to_faces": vertex_to_faces,
        "vertex_to_tetras": vertex_to_tetras,

        "vertex_to_simplex": np.asarray(tri.vertex_to_simplex),
        "scipy_delaunay": tri,
    }      
            
    return data

def tetra_circumsphere(v4):
    """
    """

    a, b, c, d = [np.asarray(v) for v in v4]

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

def AlphaShapeMethod(pcd: o3d.geometry.PointCloud, alpha: float, **kwargs) -> o3d.geometry.TriangleMesh:

    # Construct Delaunay complex
    data = prepare_delaunay_data(pcd)

    tetras = []

    for tid, tet in enumerate(data['tetras']):
        v4 = data['vertices'][tet]
        center, radius, valid = tetra_circumsphere(v4)

        if not valid:
            continue

        if radius <= alpha:
            tetras.append(tid)


    face_count = {}

    for tid in tetras:
        for fid in data["tetra_to_faces"][tid]:
            face_count[fid] = face_count.get(fid, 0) + 1

    boundary_face_ids = [fid for fid, count in face_count.items() if count == 1]
    boundary_faces = data["triangles"][boundary_face_ids]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(boundary_faces)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()

    return mesh

if __name__ == "__main__":
    pcd = BunnyPCD()
    generated_mesh = AlphaShapeMethod(pcd, alpha=0.5)
    ShowComparison(pcd, generated_mesh, BuiltinSurfaceReconstructionMethod.ALPHA_SHAPE, alpha=0.5)
