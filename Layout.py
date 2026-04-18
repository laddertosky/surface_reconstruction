import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore # Open3d doesn't provide signature for this python binding
import open3d.visualization.rendering as rendering  # type: ignore # Open3d doesn't provide signature for this python binding

from Assets import ALL_ASSETS
# TODO: replace the implementations after other team member finish their work
from Fake import AlphaShapeMethod, BallPivotingMethod, PoissonMethod

# from AlphaShape import AlphaShapeMethod
# from BallPivoting import BallPivotingMethod
# from Poisson import PoissonMethod

TOP_BAR_HEIGHT = 40
CTRL_HEIGHT    = 40
BOTTOM_HEIGHT  = 420
COLOR1 = [0.6, 0.4, 0.4, 1.0]
COLOR2 = [0.4, 0.6, 0.4, 1.0]
COLOR3 = [0.6, 0.4, 0.6, 1.0]
COLOR4 = [0.3, 0.3, 0.3, 1.0]
COLORS = [COLOR1, COLOR2, COLOR3, COLOR4]

class LayoutMode(Enum):
    All = auto()
    AlphaShapeFocused = auto()
    BallPivotFocused = auto()
    PossionFocused = auto()
    Reference = auto()

@dataclass
class CameraState:
    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray

    def __post_init__(self):
        self.forward = self.forward / np.linalg.norm(self.forward)
        self.up  = self.up  / np.linalg.norm(self.up)

    def copy(self):
        return CameraState(
            position=self.position.copy(),
            forward=self.forward.copy(),
            up=self.up.copy(),
        )

class Panel:
    def __init__(self, name: str, window: gui.Window, id: int = 1):
        self._name = name

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(window.renderer)
        self.scene_widget.scene.set_background(COLORS[id-1])
        self.scene_widget.scene.show_axes(True)

        em = window.theme.font_size
        self.control_panel = gui.Horiz(int(0.4 * em), gui.Margins(em, int(0.3 * em), em, int(0.3 * em)))
        self.control_panel.add_child(gui.Label(name))
        button = gui.Button("Reset Camera")
        button.set_on_clicked(self._on_reset)
        self.control_panel.add_child(button)

        self._center = [0, 0, 0]
        self._extent = 2.5

        self.rendered = None
        self.last_camera_state = CameraState(np.zeros(3), np.array([0, 0, 1]), np.array([0, 1, 0]))

    # should run in the UI thread, so don't put any heavy computing task here
    def add_mesh(self, mesh: o3d.geometry.TriangleMesh, aabb: o3d.geometry.AxisAlignedBoundingBox) -> None:
        self.scene_widget.scene.clear_geometry()
        self.aabb = aabb 
        self._center = aabb.get_center()
        self._extent = aabb.get_max_extent()

        if mesh.has_vertex_colors:
            mesh.vertex_colors = o3d.utility.Vector3dVector([])

        material = rendering.MaterialRecord()
        material.base_reflectance = 0.7
        material.shader = "defaultLit"

        self.scene_widget.scene.add_geometry(f"{self._name}_aabb", aabb, material)
        self.scene_widget.scene.add_geometry(f"{self._name}_mesh", mesh, material)
        print(f"{self._name}, triangles: {len(mesh.triangles)}")

    def reset_camera(self) -> None:
        self._on_reset()

    def get_camera_state(self) -> CameraState:
        camera = self.scene_widget.scene.camera

        view_matrix = np.array(camera.get_view_matrix())  # shape (4, 4)
        if np.any(np.isnan(view_matrix)):
            return self.last_camera_state

        R = view_matrix[:3, :3]

        translation = view_matrix[:3, 3]
        position = -R.T @ translation

        # sometimes it will explode
        if np.any(~np.isfinite(position)) or np.linalg.norm(position) > 1e6:
            return self.last_camera_state

        forward = -R[2]
        up = R[1]
        return CameraState(position, forward, up)

    def sync_camera(self, other_panel: Self) -> None:
        other_camera_state = other_panel.get_camera_state()
        self.last_camera_state = other_camera_state.copy()

        position = other_camera_state.position
        forward = other_camera_state.forward
        up = other_camera_state.up

        self.scene_widget.scene.camera.look_at(position+forward, position, up)

    def _on_reset(self) -> None:
        camera = self.scene_widget.scene.camera
        forward = np.array([0, 0, -1.0])
        position = self._center + np.array([0, 0, 1.2 * self._extent])
        up = np.array([0, 1, 0])
        new_camera_state = CameraState(position, forward, up)

        camera.look_at(
            self._center, 
            new_camera_state.position,
            new_camera_state.up
        )

        fov = 60
        rect = self.scene_widget.frame
        aspect = rect.width / rect.height
        camera.set_projection(
            fov,
            aspect,
            0.1, # near
            10000, # far
            rendering.Camera.FovType.Vertical
        )

class Window:
    def __init__(self, app: gui.Application, name: str, width: int, height: int):
        self.app = app
        self._window = app.create_window(name, width, height)

        self._layout_mode = LayoutMode.All
        self._camera_synced = True

        self._asset_index = 0
        self._pcd = ALL_ASSETS[self._asset_index].load_pcd()
        self._radius = ALL_ASSETS[self._asset_index].init_radius
        self._poisson_depth = 8

        self._alpha = ALL_ASSETS[self._asset_index].init_alpha
        self._alpha_debounce_delay = 0.1
        self._alpha_debounce_timer = None
        self._radii_debounce_delay = 0.3
        self._radii_debounce_timer = None
        self._depth_debounce_delay = 0.3
        self._depth_debounce_timer = None

        self._init_layout()
        self._window.set_on_layout(self._on_layout)

        self._camera_syncing_lock = False
        self._window.set_on_tick_event(self._on_tick)

        self._make_meshes(require_reset_camera=True)

    # TODO: UI thread is somehow blocked with slow mesh processing
    def _make_mesh_async(self, mode, fn, require_reset_camera: bool):
        if self._panels[mode].rendered == mode:
            return

        print(f"Preparing new mesh for {mode}")
        mesh = fn()
        mesh.compute_vertex_normals()
        mesh.orient_triangles()

        aabb = ALL_ASSETS[self._asset_index].aabb
        self._panels[mode].rendered = mode
        def _update():
            self._panels[mode].add_mesh(mesh, aabb)
            self._window.post_redraw()
            if require_reset_camera:
                self._panels[mode].reset_camera()
        self.app.post_to_main_thread(self._window, _update)

    def _make_reference_mesh(self) -> o3d.geometry.TriangleMesh:
        return ALL_ASSETS[self._asset_index].mesh

    def _make_possion_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = PoissonMethod(
            pcd=self._pcd, 
            depth=self._poisson_depth,
            # scale=1.1,
            # linear_fit=True,
        )

        aabb = ALL_ASSETS[self._asset_index].aabb
        mesh = mesh.crop(aabb)
        return mesh

    def _make_alpha_shape_mesh(self) -> o3d.geometry.TriangleMesh:
        return AlphaShapeMethod(
            pcd=self._pcd, 
            alpha=self._alpha,
        )

    def _make_ball_pivoting_mesh(self) -> o3d.geometry.TriangleMesh:
        return BallPivotingMethod(
            pcd=self._pcd, 
            radii=o3d.utility.DoubleVector([self._radius, self._radius * 2, self._radius * 4])
        )

    def _make_meshes(self, require_reset_camera: bool) -> None:
        tasks = {
            LayoutMode.Reference: self._make_reference_mesh,
            LayoutMode.PossionFocused: self._make_possion_mesh,
            LayoutMode.AlphaShapeFocused: self._make_alpha_shape_mesh,
            LayoutMode.BallPivotFocused: self._make_ball_pivoting_mesh,
        }

        if self._layout_mode == LayoutMode.All:
            for mode, fn in tasks.items():
                t = threading.Thread(target=self._make_mesh_async, args=(mode, fn, require_reset_camera), daemon=True)
                t.start()
        else:
            t1 = threading.Thread(target=self._make_mesh_async, args=(self._layout_mode, tasks[self._layout_mode], require_reset_camera), daemon=True)
            t1.start()
            t2 = threading.Thread(target=self._make_mesh_async, args=(LayoutMode.Reference, tasks[LayoutMode.Reference], require_reset_camera), daemon=True)
            t2.start()

    def _apply_alpha_change(self) -> None:
        self._panels[LayoutMode.AlphaShapeFocused].rendered = None
        self._make_meshes(require_reset_camera=False)

    def _on_alpha_changed(self, log_alpha: float) -> None:
        if self._alpha_debounce_timer:
            self._alpha_debounce_timer.cancel()

        self._alpha_debounce_timer = threading.Timer(
            self._alpha_debounce_delay,
            self._apply_alpha_change
        )
        self._alpha = np.pow(10, log_alpha)
        self._alpha_debounce_timer.start()

    def _apply_depth_change(self):
        self._panels[LayoutMode.PossionFocused].rendered = None
        self._make_meshes(require_reset_camera=False)

    # it always sends float value even if it is attached to an integer slider
    def _on_depth_changed(self, depth: float) -> None:
        if self._depth_debounce_timer:
            self._depth_debounce_timer.cancel()

        self._depth_debounce_timer = threading.Timer(
            self._depth_debounce_delay,
            self._apply_depth_change
        )
        self._poisson_depth = int(depth)
        self._depth_debounce_timer.start()

    def _apply_radius_change(self) -> None:
        self._panels[LayoutMode.BallPivotFocused].rendered = None
        self._make_meshes(require_reset_camera=False)

    def _on_radius_changed(self, radius: float) -> None:
        if self._radii_debounce_timer:
            self._radii_debounce_timer.cancel()

        self._radii_debounce_timer = threading.Timer(
            self._radii_debounce_delay,
            self._apply_radius_change
        )
        self._radius = radius
        self._radii_debounce_timer.start()

    def _init_layout(self) -> None:
        rect = self._window.content_rect
        em = self._window.theme.font_size
        top_bar = gui.Horiz(int(0.5 * em), gui.Margins(em, int(0.3 * em), em, int(0.3 * em)))
        top_bar.frame = gui.Rect(rect.x, rect.y, rect.width, TOP_BAR_HEIGHT)
        self._window.add_child(top_bar)

        all_btn = gui.Button("Show All")
        all_btn.set_on_clicked(lambda: self._set_focus(LayoutMode.All))
        top_bar.add_child(all_btn)

        possion_panel = Panel("Possion", self._window, 1)
        self._poisson_depth_slider = gui.Slider(gui.Slider.Type.INT)
        self._poisson_depth_slider.int_value = self._poisson_depth
        self._poisson_depth_slider.set_limits(2, 12)
        self._poisson_depth_slider.set_on_value_changed(self._on_depth_changed)
        possion_panel.control_panel.add_child(gui.Label("depth"))
        possion_panel.control_panel.add_child(self._poisson_depth_slider)

        alpha_shape_panel = Panel("Alpha Shape", self._window, 2)
        self._alpha_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self._alpha_slider.double_value = np.log10(self._alpha)
        self._alpha_slider.set_limits(-3, 3)
        self._alpha_slider.set_on_value_changed(self._on_alpha_changed)
        alpha_shape_panel.control_panel.add_child(gui.Label("log10(alpha)"))
        alpha_shape_panel.control_panel.add_child(self._alpha_slider)
        
        ball_pivoting_panel = Panel("Ball Pivoting", self._window, 3)
        self._ball_radii_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self._ball_radii_slider.double_value = self._radius
        self._ball_radii_slider.set_limits(0.001, 2)
        self._ball_radii_slider.set_on_value_changed(self._on_radius_changed)
        ball_pivoting_panel.control_panel.add_child(gui.Label("radius"))
        ball_pivoting_panel.control_panel.add_child(self._ball_radii_slider)

        reference_panel = Panel("Reference", self._window, 4)
        self._panels = {
            LayoutMode.AlphaShapeFocused: alpha_shape_panel,
            LayoutMode.BallPivotFocused: ball_pivoting_panel,
            LayoutMode.PossionFocused: possion_panel,
            LayoutMode.Reference:reference_panel,
        }

        content_top = rect.y + TOP_BAR_HEIGHT
        content_height = rect.height - TOP_BAR_HEIGHT - BOTTOM_HEIGHT
        slot_width = rect.width / 3
        index = 0
        for mode, panel in self._panels.items():
            if mode == LayoutMode.Reference: continue

            btn = gui.Button(f"Focus {panel._name}")
            btn.set_on_clicked(lambda m=mode: self._set_focus(m))
            top_bar.add_child(btn)

            left = rect.x + index * slot_width
            panel.control_panel.frame = gui.Rect(left, content_top, slot_width, CTRL_HEIGHT)
            panel.scene_widget.frame = gui.Rect(left, content_top + CTRL_HEIGHT, slot_width, content_height - CTRL_HEIGHT)
            self._window.add_child(panel.control_panel)
            self._window.add_child(panel.scene_widget)
            index += 1

        reference_top = content_top + content_height
        reference_panel.control_panel.frame = gui.Rect(rect.x, reference_top, rect.width, CTRL_HEIGHT)
        reference_panel.scene_widget.frame = gui.Rect(rect.x, reference_top + CTRL_HEIGHT, rect.width, BOTTOM_HEIGHT - CTRL_HEIGHT)

        self._window.add_child(reference_panel.control_panel)
        self._window.add_child(reference_panel.scene_widget)

        sync_checkbox = gui.Checkbox("Sync Cameras")
        sync_checkbox.checked = self._camera_synced
        sync_checkbox.set_on_checked(self._on_sync)
        top_bar.add_child(sync_checkbox)

        mesh_selection = gui.Combobox()
        for asset in ALL_ASSETS:
            mesh_selection.add_item(f"{asset.name} ({asset.vertices_count} vertices)")
        mesh_selection.selected_index = self._asset_index
        mesh_selection.set_on_selection_changed(self._on_selection_changed)
        top_bar.add_child(mesh_selection)

    def _on_selection_changed(self, _: str, index: int) -> None:
        if self._asset_index == index:
            return

        self._asset_index = index
        self._pcd = ALL_ASSETS[self._asset_index].load_pcd()
        self._radius = ALL_ASSETS[self._asset_index].init_radius
        self._ball_radii_slider.double_value = self._radius

        self._alpha = ALL_ASSETS[self._asset_index].init_alpha
        self._alpha_slider.double_value = np.log10(self._alpha)

        for mode, panel in self._panels.items():
            if self._layout_mode == LayoutMode.All:
                panel.rendered = None
            elif mode == LayoutMode.Reference:
                panel.rendered = None
            # defer rendering other panels if focus on a specific panel
            elif mode == self._layout_mode:
                panel.rendered = None

        self._make_meshes(require_reset_camera=True)

    def _on_tick(self) -> bool:
        if not self._camera_synced or self._camera_syncing_lock:
            return False

        def cameras_equal(previous_state: CameraState, current_state: CameraState, tol: float = 1e-3) -> bool:
            return (
                np.allclose(previous_state.position, current_state.position, atol=tol) and 
                np.allclose(previous_state.forward, current_state.forward, atol=tol)
            )

        for mode, panel in self._panels.items():
            current_camara_state = panel.get_camera_state()
            if not cameras_equal(panel.last_camera_state, current_camara_state):
                self._camera_syncing_lock = True
                for other_mode, other_panel in self._panels.items():
                    if other_mode == mode: continue
                    other_panel.sync_camera(panel)
                    panel.last_camera_state = current_camara_state.copy()

                self._camera_syncing_lock = False
                return True # found one moving is enough
        return False

    def _on_sync(self, checked) -> None:
        self._camera_synced = checked

    def _set_focus(self, mode: LayoutMode) -> None:
        self._layout_mode = mode
        if mode != LayoutMode.All and self._panels[mode].rendered != self._panels[LayoutMode.Reference].rendered:
            self._panels[mode].rendered = None

        self._make_meshes(require_reset_camera=False)
        self._window.set_needs_layout()

    def _on_layout(self, _) -> None:
        rect = self._window.content_rect
        content_top = rect.y + TOP_BAR_HEIGHT
        content_height = rect.height - TOP_BAR_HEIGHT - BOTTOM_HEIGHT

        if self._layout_mode == LayoutMode.All:
            slot_width = rect.width / 3
            index = 0
            for mode, panel in self._panels.items():
                if mode == LayoutMode.Reference: continue

                left = rect.x + slot_width * index
                panel.control_panel.frame = gui.Rect(left, content_top, slot_width, CTRL_HEIGHT)
                panel.scene_widget.frame = gui.Rect(left, content_top + CTRL_HEIGHT, slot_width, content_height - CTRL_HEIGHT)

                panel.control_panel.visible = True
                panel.scene_widget.visible = True
                index += 1

        else:
            for mode, panel in self._panels.items():
                if mode == LayoutMode.Reference: continue

                if mode == self._layout_mode:
                    panel.control_panel.frame = gui.Rect(rect.x, content_top, rect.width, CTRL_HEIGHT)
                    panel.scene_widget.frame = gui.Rect(rect.x, content_top + CTRL_HEIGHT, rect.width, content_height - CTRL_HEIGHT)
                    panel.control_panel.visible = True
                    panel.scene_widget.visible = True
                else:
                    panel.control_panel.visible = False
                    panel.scene_widget.visible = False
