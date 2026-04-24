import numpy as np
import open3d as o3d


class Asset():
    def __init__(self, name: str, path: str, vertices_count: int):
        self.name = name
        self.path = path
        self.mesh = None
        self.pcd = None
        self.vertices_count = vertices_count
        self.init_alpha = -1
        self.init_radius = -1

    def load_pcd(self, vertices_count: int = -1) -> o3d.geometry.PointCloud:
        if self.pcd: return self.pcd

        mesh = self._load()

        # center mesh to (0, 0, 0)
        self.aabb = mesh.get_axis_aligned_bounding_box()
        center = self.aabb.get_center()
        mesh.translate(-center)
        self.aabb.translate(-center)

        self.pcd = o3d.geometry.PointCloud()
        if vertices_count > 0 and vertices_count < self.vertices_count:
            o3d.utility.random.seed(0)
            self.pcd = mesh.sample_points_poisson_disk(vertices_count)
        else:
            self.pcd.points = mesh.vertices
            # self.pcd.normals = mesh.vertex_normals # directly using the mesh normals is cheated

        self.pcd.estimate_normals()

        # Lower k means smaller propagation steps, less likely to jump across gaps, default is 100
        # The most important steps to make Poisson better looking
        self.pcd.orient_normals_consistent_tangent_plane(k=6)

        # initial value for ball pivoting method
        distances = self.pcd.compute_nearest_neighbor_distance()
        avg_distances = np.mean(distances)
        self.init_alpha = avg_distances * 10.0
        self.init_radius = avg_distances * 1.5

        return self.pcd

    def _load(self) -> o3d.geometry.TriangleMesh:
        self.mesh = o3d.io.read_triangle_mesh(self.path)
        self.vertices_count = len(self.mesh.vertices)
        return self.mesh

# Built-in Ball Pivoting is very slow, it even struggles with beetle.obj
ALL_ASSETS = [
    Asset("suzanne", "./assets/suzanne.obj", 590), # hand-made 3d model
    Asset("beetle", "./assets/beetle.obj", 1254), # hand-made 3d model
    Asset("cow", "./assets/cow.obj", 2903), # hand-made 3d model
    Asset("teapot", "./assets/teapot.obj", 3241), # hand-made 3d model
    Asset("rocker-arm", "./assets/rocker-arm.obj", 10044), # generated with scanner
    Asset("stanford-bunny", o3d.data.BunnyMesh().path, 35947), # generated with scanner
    Asset("lucy", "./assets/lucy.obj", 49987), # generated with scanner
    Asset("dragon", "./assets/xyzrgb_dragon.obj", 124943), # generated with scanner
]

# for asset in ALL_ASSETS:
#     asset.load_pcd()
#     print(f"{asset.name}: {asset.vertices_count}")
