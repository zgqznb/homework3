from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import taichi as ti

WIDTH = 800
HEIGHT = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

POINT_RADIUS = 0.01                  # circles 的半径：像素单位
LINE_WIDTH = 2.0 / HEIGHT           # lines 的宽度：相对窗口高度
HIDDEN_POS = -10.0


@dataclass
class AppState:
    control_points: list[np.ndarray] = field(default_factory=list)
    use_bspline: bool = False
    antialias: bool = False

try:
    ti.init(arch=ti.gpu)
    print("[Info] Taichi initialized with GPU backend.")
except Exception:
    ti.init(arch=ti.cpu)
    print("[Info] GPU backend unavailable. Falling back to CPU backend.")

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# 控制点显示
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

# 控制折线显示：按 [p0, p1, p1, p2, p2, p3, ...] 方式排
line_vertices = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS - 1) * 2)

def de_casteljau(points: Sequence[np.ndarray], t: float) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("At least one control point is required.")

    work = [np.array(p, dtype=np.float32) for p in points]
    while len(work) > 1:
        work = [(1.0 - t) * work[i] + t * work[i + 1] for i in range(len(work) - 1)]
    return work[0]


def evaluate_bezier_curve(points: Sequence[np.ndarray], num_segments: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, num_segments + 1, dtype=np.float32)
    curve = np.zeros((num_segments + 1, 2), dtype=np.float32)
    for i, t in enumerate(ts):
        curve[i] = de_casteljau(points, float(t))
    return curve


def cubic_bspline_basis(u: float) -> np.ndarray:
    u2 = u * u
    u3 = u2 * u
    return np.array(
        [
            (-u3 + 3.0 * u2 - 3.0 * u + 1.0) / 6.0,
            (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0,
            (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0,
            u3 / 6.0,
        ],
        dtype=np.float32,
    )


def evaluate_cubic_bspline(points: Sequence[np.ndarray], num_segments: int) -> np.ndarray:
    if len(points) < 4:
        return np.empty((0, 2), dtype=np.float32)

    ctrl = np.asarray(points, dtype=np.float32)
    piece_count = len(ctrl) - 3
    curve = np.zeros((num_segments + 1, 2), dtype=np.float32)

    for i in range(num_segments + 1):
        global_t = (i / num_segments) * piece_count
        seg_id = min(int(global_t), piece_count - 1)
        local_t = global_t - seg_id

        if i == num_segments:
            seg_id = piece_count - 1
            local_t = 1.0

        b = cubic_bspline_basis(local_t)
        p = ctrl[seg_id: seg_id + 4]
        curve[i] = b[0] * p[0] + b[1] * p[1] + b[2] * p[2] + b[3] * p[3]

    return curve

@ti.func
def clamp01(x):
    return ti.max(0.0, ti.min(1.0, x))


@ti.func
def clamp_vec3(v):
    return ti.Vector([clamp01(v[0]), clamp01(v[1]), clamp01(v[2])])


@ti.func
def blend_curve_pixel(x, y, weight):
    if 0 <= x and x < WIDTH and 0 <= y and y < HEIGHT and weight > 0.0:
        curve_color = ti.Vector([0.15, 0.92, 0.28])
        pixels[x, y] = clamp_vec3(pixels[x, y] + curve_color * weight)


@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.05, 0.05, 0.08])


@ti.kernel
def draw_curve_kernel(n: ti.i32, antialias: ti.i32):
    for i in range(n):
        point = curve_points_field[i]
        px = point[0] * (WIDTH - 1)
        py = point[1] * (HEIGHT - 1)

        if antialias == 0:
            x = ti.cast(px, ti.i32)
            y = ti.cast(py, ti.i32)
            blend_curve_pixel(x, y, 1.0)
        else:
            base_x = ti.cast(ti.floor(px), ti.i32)
            base_y = ti.cast(ti.floor(py), ti.i32)

            for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
                x = base_x + dx
                y = base_y + dy
                cx = ti.cast(x, ti.f32) + 0.5
                cy = ti.cast(y, ti.f32) + 0.5
                dist = ti.sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy))
                weight = ti.max(0.0, 1.25 - dist)
                blend_curve_pixel(x, y, weight)


def pack_control_points(points: Sequence[np.ndarray]) -> np.ndarray:
    packed = np.full((MAX_CONTROL_POINTS, 2), HIDDEN_POS, dtype=np.float32)
    if len(points) > 0:
        packed[: len(points)] = np.asarray(points, dtype=np.float32)
    return packed


def pack_line_vertices(points: Sequence[np.ndarray]) -> np.ndarray:
    packed = np.full(((MAX_CONTROL_POINTS - 1) * 2, 2), HIDDEN_POS, dtype=np.float32)

    line_count = min(max(len(points) - 1, 0), MAX_CONTROL_POINTS - 1)
    for i in range(line_count):
        packed[2 * i] = points[i]
        packed[2 * i + 1] = points[i + 1]

    return packed


def sync_gui_fields(points: Sequence[np.ndarray]) -> None:
    gui_points.from_numpy(pack_control_points(points))
    line_vertices.from_numpy(pack_line_vertices(points))


def build_curve(points: Sequence[np.ndarray], use_bspline: bool) -> np.ndarray:
    if use_bspline:
        return evaluate_cubic_bspline(points, NUM_SEGMENTS)
    if len(points) >= 2:
        return evaluate_bezier_curve(points, NUM_SEGMENTS)
    return np.empty((0, 2), dtype=np.float32)


def mode_name(state: AppState) -> str:
    return "B-spline" if state.use_bspline else "Bezier"


def print_status(state: AppState) -> None:
    print(
        f"[Status] mode={mode_name(state)}, "
        f"antialias={'ON' if state.antialias else 'OFF'}, "
        f"control_points={len(state.control_points)}"
    )



def handle_events(window: ti.ui.Window, state: AppState) -> bool:
    should_quit = False

    while window.get_event(ti.ui.PRESS):
        key = window.event.key

        if key == ti.ui.ESCAPE:
            should_quit = True
        elif key == ti.ui.LMB:
            if len(state.control_points) < MAX_CONTROL_POINTS:
                pos = np.array(window.get_cursor_pos(), dtype=np.float32)
                state.control_points.append(np.clip(pos, 0.0, 1.0))
                print_status(state)
        else:
            key_str = str(key).lower()
            if key_str == "c":
                state.control_points.clear()
                print_status(state)
            elif key_str == "b":
                state.use_bspline = not state.use_bspline
                print_status(state)
            elif key_str == "a":
                state.antialias = not state.antialias
                print_status(state)

    return should_quit


def main() -> None:
    state = AppState()
    window = ti.ui.Window("Bezier / B-spline Curve Lab", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()

    print("[Help] Left click: add control point | C: clear | B: toggle B-spline | A: toggle antialias | ESC: quit")
    print_status(state)

    while window.running:
        if handle_events(window, state):
            break

        clear_pixels()
        sync_gui_fields(state.control_points)

        curve = build_curve(state.control_points, state.use_bspline)
        if len(curve) > 0:
            curve_points_field.from_numpy(curve)
            draw_curve_kernel(curve.shape[0], 1 if state.antialias else 0)

        canvas.set_image(pixels)
        canvas.lines(line_vertices, LINE_WIDTH, color=(0.72, 0.72, 0.72))
        canvas.circles(gui_points, POINT_RADIUS, color=(0.95, 0.22, 0.22))
        window.show()


if __name__ == "__main__":
    main()