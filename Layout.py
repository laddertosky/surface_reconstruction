import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui  # type: ignore # Open3d doesn't provide signature for this python binding
import open3d.visualization.rendering as rendering  # type: ignore # Open3d doesn't provide signature for this python binding

from AlphaShape import AlphaShapeMethod
from Assets import ALL_ASSETS
from BallPivoting import BallPivotingMethod
from Poisson import PoissonMethod

TOP_BAR_HEIGHT = 40
CTRL_HEIGHT    = 40
BOTTOM_HEIGHT  = 420
COLOR1 = [0.6, 0.4, 0.4, 1.0]
COLOR2 = [0.4, 0.6, 0.4, 1.0]
COLOR3 = [0.6, 0.4, 0.6, 1.0]
COLOR4 = [0.3, 0.3, 0.3, 1.0]
COLORS = [COLOR1, COLOR2, COLOR3, COLOR4]
FOV = 60

class LayoutMode(Enum):
    All = auto()
    AlphaShapeFocused = auto()
    BallPivotFocused = auto()
    PossionFocused = auto()
    Reference = auto()

@dataclass
class CameraState:
    position: np.ndarray
    up: np.ndarray

    def __post_init__(self):
        self.up  = self.up  / np.linalg.norm(self.up)

    def copy(self):
        return CameraState(
            position=self.position.copy(),
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

        self._center = np.zeros(3)
        self._extent = 2.5

        self._material = rendering.MaterialRecord()
        self._material.base_reflectance = 0.7
        self._material.shader = "defaultLit"

        self.rendered = -1
        self.last_camera_state = CameraState(position=self._center, up=np.array([0, 1, 0]))

    # should run in the UI thread, so don't put any heavy computing task here
    def add_mesh(self, mesh: o3d.geometry.TriangleMesh, aabb: o3d.geometry.AxisAlignedBoundingBox) -> None:
        self.scene_widget.scene.clear_geometry()
        self._center = aabb.get_center()
        self._extent = aabb.get_max_extent()

        if mesh.has_vertex_colors:
            mesh.vertex_colors = o3d.utility.Vector3dVector([])

        self.scene_widget.scene.add_geometry(f"{self._name}_aabb", aabb, self._material)
        self.scene_widget.scene.add_geometry(f"{self._name}_mesh", mesh, self._material)
        print(f"{self._name}, triangles: {len(mesh.triangles)}")

    def reset_camera(self) -> None:
        self._on_reset()

    def get_camera_state(self) -> CameraState:
        camera = self.scene_widget.scene.camera

        model_matrix = np.array(camera.get_model_matrix())  # shape (4, 4)
        if np.any(np.isnan(model_matrix)):
            return self.last_camera_state

        position = model_matrix[:3, 3]
        up = model_matrix[:3, 1]
        return CameraState(position, up)

    def sync_camera(self, other_panel: Self) -> None:
        other_aabb = other_panel.scene_widget.scene.bounding_box
        other_camera_state = other_panel.get_camera_state()
        self.last_camera_state = other_camera_state.copy()

        position = other_camera_state.position
        up = other_camera_state.up

        # reset rotation pivot even after right mouse drag
        self.scene_widget.setup_camera(FOV, other_aabb, self._center)
        self.scene_widget.scene.camera.look_at(self._center, position, up)

    def _on_reset(self) -> None:
        aabb = self.scene_widget.scene.bounding_box
        # reset rotation pivot even after right mouse drag
        self.scene_widget.setup_camera(FOV, aabb, self._center) 

        camera = self.scene_widget.scene.camera
        position = self._center + np.array([0, 0, 1.2 * self._extent]) # prevent the object hit near plane during rotation
        up = np.array([0, 1, 0])
        new_camera_state = CameraState(position, up)

        camera.look_at(
            self._center, 
            new_camera_state.position,
            new_camera_state.up
        )

        rect = self.scene_widget.frame
        aspect = rect.width / rect.height
        camera.set_projection(
            FOV,
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

        self._asset_index = 3
        self._vertex_count = -1
        self._pcd = ALL_ASSETS[self._asset_index].load_pcd(self._vertex_count)
        self._radius = ALL_ASSETS[self._asset_index].init_radius
        self._poisson_depth = 6

        self._alpha = ALL_ASSETS[self._asset_index].init_alpha
        self._alpha_debounce_delay = 0.1
        self._alpha_debounce_timer = None
        self._radii_debounce_delay = 0.3
        self._radii_debounce_timer = None
        self._depth_debounce_delay = 0.3
        self._depth_debounce_timer = None
        self._threads = {}

        self._init_layout()
        self._window.set_on_layout(self._on_layout)

        self._camera_syncing_lock = False
        self._window.set_on_tick_event(self._on_tick)

        self._make_meshes(require_reset_camera=True)

    def _make_mesh_async(self, mode, asset_index, fn, require_reset_camera: bool):
        if self._panels[mode].rendered == asset_index:
            return

        self._panels[mode].rendered = asset_index

        print(f"Preparing new mesh for {mode}")
        mesh = fn()
        mesh.compute_vertex_normals()
        mesh.orient_triangles()

        aabb = ALL_ASSETS[self._asset_index].aabb

        def _update():
            self._panels[mode].add_mesh(mesh, aabb)
            self._window.post_redraw()
            if require_reset_camera:
                self._panels[mode].reset_camera()
        self.app.post_to_main_thread(self._window, _update)

    def _make_reference_mesh(self) -> o3d.geometry.TriangleMesh:
        return ALL_ASSETS[self._asset_index].mesh

    def _make_possion_mesh(self) -> o3d.geometry.TriangleMesh:
        print(f"Poisson with depth = {self._poisson_depth}")

        return PoissonMethod(
            pcd=self._pcd, 
            depth=self._poisson_depth,
        )


    def _make_alpha_shape_mesh(self) -> o3d.geometry.TriangleMesh:
        print(f"Alpha Shape with alpha = {self._alpha:>.3f}")
        return AlphaShapeMethod(
            pcd=self._pcd, 
            alpha=self._alpha,
        )

    def _make_ball_pivoting_mesh(self) -> o3d.geometry.TriangleMesh:
        radius = self._radius
        radii = o3d.utility.DoubleVector([radius, radius * 2, radius * 4])
        print(f"Ball Pivoting with radius = {radii}")
        mesh = BallPivotingMethod(
            pcd=self._pcd, 
            radii=radii
        )

        # use the previous local value to update slider display value
        self._ball_radii_slider.double_value = radius

        return mesh

    def _make_meshes(self, require_reset_camera: bool) -> None:
        tasks = {
            LayoutMode.Reference: self._make_reference_mesh,
            LayoutMode.PossionFocused: self._make_possion_mesh,
            LayoutMode.AlphaShapeFocused: self._make_alpha_shape_mesh,
            LayoutMode.BallPivotFocused: self._make_ball_pivoting_mesh,
        }

        if self._layout_mode == LayoutMode.All:
            for mode, fn in tasks.items():
                t = self._threads.get(mode, None)
                if t and t.is_alive():
                    continue

                t = threading.Thread(target=self._make_mesh_async, args=(mode, self._asset_index, fn, require_reset_camera), daemon=True)
                t.start()
                self._threads[mode] = t
        else:
            t = self._threads.get(self._layout_mode, None)
            if not t or not t.is_alive():
                t = threading.Thread(target=self._make_mesh_async, args=(self._layout_mode, self._asset_index, tasks[self._layout_mode], require_reset_camera), daemon=True)
                t.start()
                self._threads[self._layout_mode] = t

            t = self._threads.get(LayoutMode.Reference, None)
            if not t or not t.is_alive():
                t = threading.Thread(target=self._make_mesh_async, args=(LayoutMode.Reference, self._asset_index, tasks[LayoutMode.Reference], require_reset_camera), daemon=True)
                t.start()
                self._threads[LayoutMode.Reference] = t

    def _apply_alpha_change(self) -> None:
        self._panels[LayoutMode.AlphaShapeFocused].rendered = -1
        self._make_meshes(require_reset_camera=False)

    def _on_alpha_changed(self, log_alpha: float) -> None:
        next_alpha = np.pow(10, log_alpha)
        if np.isclose(self._alpha, next_alpha):
            return

        if self._alpha_debounce_timer:
            self._alpha_debounce_timer.cancel()

        self._alpha_debounce_timer = threading.Timer(
            self._alpha_debounce_delay,
            self._apply_alpha_change
        )
        self._alpha = next_alpha
        self._alpha_debounce_timer.start()

    def _apply_depth_change(self):
        self._panels[LayoutMode.PossionFocused].rendered = -1
        self._make_meshes(require_reset_camera=False)

    # it always sends float value even if it is attached to an integer slider
    def _on_depth_changed(self, depth: float) -> None:
        if np.isclose(self._poisson_depth, depth):
            return

        if self._depth_debounce_timer:
            self._depth_debounce_timer.cancel()

        self._depth_debounce_timer = threading.Timer(
            self._depth_debounce_delay,
            self._apply_depth_change
        )
        self._poisson_depth = round(depth)
        self._depth_debounce_timer.start()

    def _apply_radius_change(self) -> None:
        self._panels[LayoutMode.BallPivotFocused].rendered = -2
        self._make_meshes(require_reset_camera=False)

    def _on_radius_changed(self, radius: float) -> None:
        if np.isclose(self._radius, radius):
            return

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
        self._poisson_depth_slider.set_limits(2, 9)
        self._poisson_depth_slider.set_on_value_changed(self._on_depth_changed)
        possion_panel.control_panel.add_child(gui.Label("depth"))
        possion_panel.control_panel.add_child(self._poisson_depth_slider)

        alpha_shape_panel = Panel("Alpha Shape", self._window, 2)
        self._alpha_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self._alpha_slider.double_value = np.log10(self._alpha)
        self._alpha_slider.set_limits(-2, 0)
        self._alpha_slider.set_on_value_changed(self._on_alpha_changed)
        alpha_shape_panel.control_panel.add_child(gui.Label("log10(alpha)"))
        alpha_shape_panel.control_panel.add_child(self._alpha_slider)
        
        ball_pivoting_panel = Panel("Ball Pivoting", self._window, 3)
        self._ball_radii_slider = gui.Slider(gui.Slider.Type.DOUBLE)
        self._ball_radii_slider.double_value = self._radius
        self._ball_radii_slider.set_limits(0.0001, 0.1)
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

            self._threads[mode] = None

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
        self._pcd = ALL_ASSETS[self._asset_index].load_pcd(self._vertex_count)
        self._radius = ALL_ASSETS[self._asset_index].init_radius
        self._ball_radii_slider.double_value = self._radius

        self._alpha = ALL_ASSETS[self._asset_index].init_alpha
        self._alpha_slider.double_value = np.log10(self._alpha)

        for mode, panel in self._panels.items():
            if self._layout_mode == LayoutMode.All:
                panel.rendered = -1
            elif mode == LayoutMode.Reference:
                panel.rendered = -1
            # defer rendering other panels if focus on a specific panel
            elif mode == self._layout_mode:
                panel.rendered = -1

        self._make_meshes(require_reset_camera=True)

    def _on_tick(self) -> bool:
        if not self._camera_synced or self._camera_syncing_lock:
            return False

        def cameras_equal(previous_state: CameraState, current_state: CameraState, tol: float = 1e-3) -> bool:
            return np.allclose(previous_state.position, current_state.position, atol=tol)

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
            self._panels[mode].rendered = -1

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
