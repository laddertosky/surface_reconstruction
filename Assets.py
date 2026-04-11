import open3d as o3d


class Asset():
    def __init__(self, name: str, path: str, vertices_count: int):
        self.name = name
        self.path = path
        self.mesh = None
        self.pcd = None
        self.vertices_count = vertices_count

    def load_pcd(self, vertices_count: int = -1) -> o3d.geometry.PointCloud:
        if self.pcd: return self.pcd

        mesh = self._load()

        self.pcd = o3d.geometry.PointCloud()
        if vertices_count > 0 and vertices_count < self.vertices_count:
            o3d.utility.random.seed(0)
            self.pcd = mesh.sample_points_poisson_disk(vertices_count)
        else:
            self.pcd.points = mesh.vertices

        if mesh.has_vertex_colors():
            self.pcd.colors = mesh.vertex_colors

        self.pcd.estimate_normals()
        return self.pcd

    def _load(self) -> o3d.geometry.TriangleMesh:
        if self.mesh: return self.mesh

        self.mesh = o3d.io.read_triangle_mesh(self.path)
        self.vertices_count = len(self.mesh.vertices)
        return self.mesh

# Built-in Ball Pivoting is very slow, it even struggles with beetle.obj
ALL_ASSETS = [
    Asset("suzanne", "./assets/suzanne.obj", 64),
    Asset("beetle", "./assets/beetle.obj", 1254),
    Asset("cow", "./assets/cow.obj", 2903),
    Asset("teapot", "./assets/teapot.obj", 3241),
    Asset("rocket-arm", "./assets/rocker-arm.obj", 10044),
    Asset("stanford-bunny", o3d.data.BunnyMesh().path, 35947),
    Asset("lucy", "./assets/lucy.obj", 49987),
    Asset("dragon", "./assets/xyzrgb_dragon.obj", 124943),
]

for asset in ALL_ASSETS:
    asset.load_pcd()
    print(f"{asset.name}: {asset.vertices_count}")
