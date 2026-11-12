import cv2
import mediapipe as mp
import pygame
import sys
import pygame.gfxdraw
import os
import urllib.request
import json


# MediaPipe Tasks — детектор обличчя
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "models/face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def ensure_model_exists(path: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Model file not found. Downloading to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print("Model downloaded successfully.")

ensure_model_exists(model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)
FACE_MESH_EDGES = [(int(a), int(b)) for a, b in mp.solutions.face_mesh.FACEMESH_TESSELATION]
FACE_OVAL_EDGES = [(int(a), int(b)) for a, b in mp.solutions.face_mesh.FACEMESH_FACE_OVAL]

def build_face_mesh_triangles(edges):
    adjacency = {}
    for a, b in edges:
        a, b = int(a), int(b)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    triangles = set()
    for a, neighbors in adjacency.items():
        for b in neighbors:
            if b <= a:
                continue
            shared = neighbors.intersection(adjacency.get(b, set()))
            for c in shared:
                if c <= b:
                    continue
                triangles.add((a, b, c))
    return sorted(triangles)

FACE_MESH_TRIANGLES = build_face_mesh_triangles(FACE_MESH_EDGES)

cap = cv2.VideoCapture(0)
# 1280×720 — нормальна якість; на більшості вебкамер все одно ~30 FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

pygame.init()
WIDTH, HEIGHT = 980, 500
MENU_H = 48
GAP = 20
CAMERA_PANEL_W = 420
RENDER_PANEL_X = CAMERA_PANEL_W + GAP
RENDER_PANEL_W = WIDTH - CAMERA_PANEL_W - GAP
RENDER_CENTER_X = RENDER_PANEL_X + RENDER_PANEL_W // 2
RENDER_CENTER_Y = HEIGHT // 2
SETTINGS_PATH = "settings.json"
LAYOUT_PRESETS = ["50_50", "40_60", "30_70", "camera_only", "render_only"]
LAYOUT_RATIO = {
    "50_50": 0.50,
    "40_60": 0.40,
    "30_70": 0.30,
}
LAYOUT_LABELS = {
    "50_50": "50/50",
    "40_60": "40/60",
    "30_70": "30/70",
    "camera_only": "Тільки камера",
    "render_only": "Тільки рендер",
}
DEFAULT_SETTINGS = {
    "showHud": True,
    "showCameraPanel": True,
    "layoutPreset": "40_60",
    "meshScale": 1.0,
    "meshLineWidth": 2,
    "cameraMeshLineWidth": 2,
    "fpsLimit": 0,
    "qualityMode": "med",
    "meshStyle": "hybrid",
    "hudScale": 1.0,
}
FPS_LIMITS = [0, 30, 45, 60, 120]
QUALITY_MODES = ["low", "med", "high"]
MESH_STYLES = ["wire", "hybrid", "solid"]
DETECT_WIDTH_BY_QUALITY = {
    "low": 320,
    "med": 480,
    "high": 720,
}
# Пропуск кадрів між викликами детектора (1 — кожен кадр)
DETECT_STRIDE_BY_QUALITY = {
    "low": 2,
    "med": 2,
    "high": 1,
}

def normalize_settings(settings):
    normalized = dict(DEFAULT_SETTINGS)
    if isinstance(settings, dict):
        normalized.update(settings)
    normalized["showHud"] = bool(normalized.get("showHud", True))
    normalized["showCameraPanel"] = bool(normalized.get("showCameraPanel", True))
    layout_preset = str(normalized.get("layoutPreset", "40_60"))
    if layout_preset not in LAYOUT_PRESETS:
        layout_preset = "40_60"
    normalized["layoutPreset"] = layout_preset
    try:
        normalized["meshScale"] = float(normalized.get("meshScale", 1.0))
    except (TypeError, ValueError):
        normalized["meshScale"] = 1.0
    normalized["meshScale"] = max(0.6, min(2.2, normalized["meshScale"]))
    try:
        normalized["meshLineWidth"] = int(normalized.get("meshLineWidth", 2))
    except (TypeError, ValueError):
        normalized["meshLineWidth"] = 2
    normalized["meshLineWidth"] = max(1, min(5, normalized["meshLineWidth"]))
    try:
        normalized["cameraMeshLineWidth"] = int(normalized.get("cameraMeshLineWidth", 2))
    except (TypeError, ValueError):
        normalized["cameraMeshLineWidth"] = 2
    normalized["cameraMeshLineWidth"] = max(1, min(8, normalized["cameraMeshLineWidth"]))
    try:
        normalized["fpsLimit"] = int(normalized.get("fpsLimit", 0))
    except (TypeError, ValueError):
        normalized["fpsLimit"] = 0
    if normalized["fpsLimit"] not in FPS_LIMITS:
        normalized["fpsLimit"] = 0
    quality_mode = str(normalized.get("qualityMode", "med"))
    if quality_mode not in QUALITY_MODES:
        quality_mode = "med"
    normalized["qualityMode"] = quality_mode
    mesh_style = str(normalized.get("meshStyle", "hybrid"))
    if mesh_style not in MESH_STYLES:
        mesh_style = "hybrid"
    normalized["meshStyle"] = mesh_style
    try:
        normalized["hudScale"] = float(normalized.get("hudScale", 1.0))
    except (TypeError, ValueError):
        normalized["hudScale"] = 1.0
    normalized["hudScale"] = max(0.8, min(2.0, normalized["hudScale"]))
    return normalized

def load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return normalize_settings(data)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(DEFAULT_SETTINGS)

def save_settings(settings):
    safe_settings = normalize_settings(settings)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(safe_settings, f, ensure_ascii=True, indent=2)

runtime_settings = load_settings()

def recalc_layout():
    global CAMERA_PANEL_W, RENDER_PANEL_X, RENDER_PANEL_W, RENDER_CENTER_X, RENDER_CENTER_Y
    preset = runtime_settings["layoutPreset"]
    if preset == "camera_only":
        # Лише прев’ю камери
        CAMERA_PANEL_W = WIDTH
        RENDER_PANEL_X = WIDTH
        RENDER_PANEL_W = 0
    elif preset == "render_only":
        # Лише панель рендеру
        CAMERA_PANEL_W = 0
        RENDER_PANEL_X = 0
        RENDER_PANEL_W = WIDTH
    elif runtime_settings["showCameraPanel"]:
        ratio = LAYOUT_RATIO.get(preset, 0.4)
        CAMERA_PANEL_W = max(220, int(WIDTH * ratio))
        if CAMERA_PANEL_W > WIDTH - GAP - 220:
            CAMERA_PANEL_W = max(220, WIDTH - GAP - 220)
        RENDER_PANEL_X = CAMERA_PANEL_W + GAP
        RENDER_PANEL_W = max(220, WIDTH - CAMERA_PANEL_W - GAP)
    else:
        CAMERA_PANEL_W = 0
        RENDER_PANEL_X = 0
        RENDER_PANEL_W = WIDTH
    RENDER_CENTER_Y = MENU_H + (HEIGHT - MENU_H) // 2
    RENDER_CENTER_X = RENDER_PANEL_X + RENDER_PANEL_W // 2

recalc_layout()
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Mirror Avatar")
clock = pygame.time.Clock()

smoothed_params = {}
alpha_face = 0.2 

def get_face_params(blendshapes):
    
    if not blendshapes:
        return None
    
    s = {b.category_name: b.score for b in blendshapes}
    
    return {
        "eye_l_open": 1.0 - s.get("eyeBlinkRight", 0),   # 1 — відкрито
        "eye_r_open": 1.0 - s.get("eyeBlinkLeft", 0),
        "squint": (s.get("eyeSquintLeft", 0) + s.get("eyeSquintRight", 0)) / 2,
        "brow_l_up": s.get("browOuterUpRight", 0) - s.get("browDownRight", 0),
        "brow_r_up": s.get("browOuterUpLeft", 0) - s.get("browDownLeft", 0),
        "nose_sneer": (s.get("noseSneerLeft", 0) + s.get("noseSneerRight", 0)) / 2,
        "mouth_open": s.get("jawOpen", 0),
        "mouth_smile": (s.get("mouthSmileLeft", 0) + s.get("mouthSmileRight", 0)) / 2,
        "mouth_frown": (s.get("mouthFrownLeft", 0) + s.get("mouthFrownRight", 0)) / 2,
        "mouth_pucker": s.get("mouthPucker", 0),
        "mouth_wide": s.get("mouthStretchLeft", 0) + s.get("mouthStretchRight", 0),
        "tongue_out": s.get("tongueOut", 0),
        # погляд: мінус X — ліворуч, мінус Y — вгору
        "eye_l_gaze_x": s.get("eyeLookInLeft", 0) - s.get("eyeLookOutLeft", 0),
        "eye_l_gaze_y": s.get("eyeLookDownLeft", 0) - s.get("eyeLookUpLeft", 0),
        "eye_r_gaze_x": s.get("eyeLookOutRight", 0) - s.get("eyeLookInRight", 0),
        "eye_r_gaze_y": s.get("eyeLookDownRight", 0) - s.get("eyeLookUpRight", 0),
    }

def get_emotion_scores(p, neutral_baseline=None):
    # Прості евристики; baseline — «спокійне» обличчя користувача
    if neutral_baseline is None:
        neutral_baseline = {}

    def cal(name):
        return max(0.0, p.get(name, 0.0) - neutral_baseline.get(name, 0.0))

    def cal_signed(name):
        return p.get(name, 0.0) - neutral_baseline.get(name, 0.0)

    smile = cal("mouth_smile")
    frown = cal("mouth_frown")
    mouth_open = cal("mouth_open")
    pucker = cal("mouth_pucker")
    squint = cal("squint")
    nose_sneer = cal("nose_sneer")
    brow_l = cal_signed("brow_l_up")
    brow_r = cal_signed("brow_r_up")
    brow_avg = (brow_l + brow_r) * 0.5

    scores = {
        "Happy": max(0.0, smile * 1.2 - frown * 0.4),
        "Sad": max(0.0, frown * 1.35 + max(0.0, -brow_avg) * 0.35 - smile * 0.35 - pucker * 0.55),
        "Surprised": max(0.0, mouth_open * 0.9 + max(0.0, brow_avg) * 0.7),
        "Excited": max(0.0, smile * 0.8 + mouth_open * 0.8 + max(0.0, brow_avg) * 0.4),
        "Angry": max(0.0, frown * 1.25 + max(0.0, -brow_avg) * 1.1 + squint * 0.7 + nose_sneer * 0.65),
        "Focused": max(0.0, squint * 1.1 + max(0.0, -brow_avg) * 0.5),
        "Disgusted": max(0.0, nose_sneer * 1.2 + frown * 0.6 + squint * 0.4),
        "Thinking": max(0.0, max(0.0, -brow_avg) * 0.9 + pucker * 0.4 + mouth_open * 0.2),
        "Duck Face": max(0.0, pucker * 1.55 - frown * 0.65 - max(0.0, -brow_avg) * 0.2),
    }
    expressive_peak = max(scores.values()) if scores else 0.0
    scores["Neutral"] = max(0.03, 0.10 - expressive_peak * 0.55)
    return scores

def update_emotion_state(p, emotion_state, neutral_baseline):
    raw_scores = get_emotion_scores(p, neutral_baseline)
    alpha = 0.18
    for key, raw in raw_scores.items():
        prev = emotion_state["smooth_scores"].get(key, 0.0)
        emotion_state["smooth_scores"][key] = prev * (1.0 - alpha) + raw * alpha

    ranked = sorted(
        emotion_state["smooth_scores"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_label, top_score = ranked[0]
    last_label = emotion_state.get("active_label", "Neutral")
    last_score = emotion_state["smooth_scores"].get(last_label, 0.0)

    # Не стрибаємо на нову емоцію без запасу; Neutral трохи «слабший»
    hysteresis_ratio = 1.12
    if last_label == "Neutral" and top_label != "Neutral":
        hysteresis_ratio = 1.02
    elif top_label == "Neutral" and last_label != "Neutral":
        hysteresis_ratio = 1.18

    if top_label != last_label and top_score < last_score * hysteresis_ratio:
        top_label, top_score = last_label, last_score

    emotion_state["active_label"] = top_label
    emotion_state["active_score"] = top_score
    emotion_state["top3"] = ranked[:3]
    return emotion_state

def estimate_head_pose(landmarks):
    # Груба оцінка повороту голови з 3D-точок
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose = landmarks[1]
    chin = landmarks[152]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]

    yaw = (left_cheek.z - right_cheek.z) * 8.0
    pitch = ((nose.y - (left_eye.y + right_eye.y) * 0.5) - ((chin.y - nose.y) * 0.35)) * 6.0
    eye_dx = max(1e-5, (right_eye.x - left_eye.x))
    roll = (right_eye.y - left_eye.y) / eye_dx

    yaw = max(-1.0, min(1.0, yaw))
    pitch = max(-1.0, min(1.0, pitch))
    roll = max(-0.7, min(0.7, roll))
    return yaw, pitch, roll

def draw_hud(screen, p, yaw, pitch, roll, has_face, fps, emotion_state):
    if p and emotion_state:
        emotion    = emotion_state.get("active_label", "Neutral")
        confidence = emotion_state.get("active_score", 0.0)
        top3       = emotion_state.get("top3", [])
    else:
        emotion, confidence = "No face", 0.0
        top3 = []

    tracking_ok = has_face
    font_sm = pygame.font.SysFont("Menlo", 14)
    font_em = pygame.font.SysFont("Menlo", 16)

    pad = 12
    line_h = 20

    em_line   = f"{emotion}  {int(confidence * 100)}%"
    top3_str  = "  ".join(f"{n}:{int(s*100)}%" for n, s in top3) if top3 else ""
    track_str = "Tracking: ON" if tracking_ok else "Tracking: LOST"
    fps_str   = f"FPS  {fps:.0f}"
    pose_str  = f"Y{yaw:+.1f}  P{pitch:+.1f}  R{roll:+.1f}"

    rows = [em_line]
    if top3_str:
        rows.append(top3_str)
    rows += ["", track_str, fps_str, pose_str]

    max_text_w = max(
        (font_em if i == 0 else font_sm).size(row)[0]
        for i, row in enumerate(rows) if row
    )
    box_w = max_text_w + pad * 2
    box_h = pad * 2 + line_h * len(rows) + 2
    box_x = WIDTH - box_w - 10
    box_y = HEIGHT - box_h - 10

    panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    panel.fill((8, 12, 20, 190))
    screen.blit(panel, (box_x, box_y))
    pygame.draw.rect(screen, (60, 120, 180), (box_x, box_y, box_w, box_h), 1, border_radius=6)

    for i, row in enumerate(rows):
        if not row:
            continue
        if i == 0:
            color = (130, 220, 255)
            surf  = font_em.render(row, True, color)
        elif "LOST" in row:
            color = (255, 150, 100)
            surf  = font_sm.render(row, True, color)
        else:
            color = (200, 215, 230)
            surf  = font_sm.render(row, True, color)
        screen.blit(surf, (box_x + pad, box_y + pad + i * line_h))

    return {}

def draw_toast(screen, message, ttl_ms):
    if not message:
        return
    now_ms = pygame.time.get_ticks()
    if now_ms > ttl_ms:
        return
    font = pygame.font.SysFont("Menlo", 18)
    text = font.render(message, True, (235, 245, 255))
    pad_x, pad_y = 14, 8
    box_w = text.get_width() + pad_x * 2
    box_h = text.get_height() + pad_y * 2
    box_x = RENDER_PANEL_X + max(10, (RENDER_PANEL_W - box_w) // 2)
    box_y = HEIGHT - box_h - 16
    panel = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    panel.fill((12, 18, 26, 210))
    screen.blit(panel, (box_x, box_y))
    pygame.draw.rect(screen, (100, 180, 230), (box_x, box_y, box_w, box_h), 1, border_radius=10)
    screen.blit(text, (box_x + pad_x, box_y + pad_y))

def project_face_landmarks(landmarks, off_x, off_y, center_x, center_y, cam_aspect, mesh_scale, panel_w=None, panel_h=None):
    if panel_w is None:
        panel_w = RENDER_PANEL_W
    if panel_h is None:
        panel_h = HEIGHT
    center_x = int(center_x - off_x * panel_w * 0.44)
    center_y = int(center_y + off_y * panel_h * 0.26)
    face_w = max(1e-5, abs(landmarks[454].x - landmarks[234].x))
    scale = min(650, max(360, 250 / face_w))
    focal = 800
    nose = landmarks[1]

    points3d = {}
    projected = {}
    for i, lm in enumerate(landmarks):
        lx = (lm.x - nose.x) * scale
        ly = (lm.y - nose.y) * scale * cam_aspect
        lz_lighting = (lm.z - nose.z) * scale * 1.4   # нормалі / світло
        lz_proj     = (lm.z - nose.z) * scale * 0.4   # перспектива
        points3d[i] = (lx, ly, lz_lighting)

        perspective = focal / max(80.0, (focal + lz_proj))
        x2d = int(center_x + lx * perspective)
        y2d = int(center_y + ly * perspective)
        projected[i] = (x2d, y2d, lz_lighting)

    xs = [v[0] for v in projected.values()]
    ys = [v[1] for v in projected.values()]
    if not xs or not ys:
        return None, None

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bbox_w = max(1, max_x - min_x)
    bbox_h = max(1, max_y - min_y)
    fit_w = panel_w * 0.58
    fit_h = panel_h * 0.80
    fit_scale = min(fit_w / bbox_w, fit_h / bbox_h) * mesh_scale

    bbox_cx = (min_x + max_x) * 0.5
    bbox_cy = min_y + (max_y - min_y) * 0.54  # центр трохи вище — більше місця для чола
    for i, (px, py, pz) in projected.items():
        px = int((px - bbox_cx) * fit_scale + center_x)
        py = int((py - bbox_cy) * fit_scale + center_y)
        projected[i] = (px, py, pz)
    for i, (lx, ly, lz) in points3d.items():
        points3d[i] = (lx * fit_scale, ly * fit_scale, lz * fit_scale)

    return projected, points3d

def draw_3d_face_mesh(
    screen,
    landmarks,
    off_x,
    off_y,
    center_x,
    center_y,
    cam_aspect,
    mesh_scale,
    mesh_line_width,
    mesh_style,
    quality_mode,
    face_fill_color=None,
    wireframe_override=None,
    panel_w=None,
    panel_h=None,
):
    projected, points3d = project_face_landmarks(
        landmarks, off_x, off_y, center_x, center_y, cam_aspect, mesh_scale, panel_w, panel_h
    )
    if not projected:
        return

    oval_pts = []
    for idx in CONTOUR_INDICES:
        if idx in projected:
            oval_pts.append((projected[idx][0], projected[idx][1]))
    if len(oval_pts) > 10 and mesh_style in ("solid", "hybrid"):
        pygame.draw.polygon(screen, (38, 40, 44), oval_pts)
        pygame.draw.aalines(screen, (90, 94, 102), True, oval_pts)

    if face_fill_color is None and mesh_style in ("solid", "hybrid"):
        face_fill_color = (210, 178, 158)

    if face_fill_color is not None and points3d:
        def normalize3(v):
            ln = max(1e-9, (v[0]*v[0]+v[1]*v[1]+v[2]*v[2])**0.5)
            return (v[0]/ln, v[1]/ln, v[2]/ln)

        key_dir  = normalize3((-0.35, -0.55,  0.76))
        fill_dir = normalize3(( 0.70, -0.20,  0.68))
        rim_dir  = normalize3(( 0.10, -0.30, -0.95))
        view_dir = (0.0, 0.0, 1.0)
        hv = normalize3((key_dir[0]+view_dir[0], key_dir[1]+view_dir[1], key_dir[2]+view_dir[2]))

        base_r, base_g, base_b = face_fill_color

        def shade_vertex(nx, ny, nz):
            ambient   = 0.38
            key_diff  = max(0.0, nx*key_dir[0]  + ny*key_dir[1]  + nz*key_dir[2])
            fill_diff = max(0.0, nx*fill_dir[0] + ny*fill_dir[1] + nz*fill_dir[2])
            rim_dot   = max(0.0, -(nx*rim_dir[0] + ny*rim_dir[1] + nz*rim_dir[2]))
            rim_str   = rim_dot**2 * 0.18
            spec_dot  = max(0.0, nx*hv[0] + ny*hv[1] + nz*hv[2])
            specular  = spec_dot**24 * 0.18
            intensity = ambient + key_diff * 0.52 + fill_diff * 0.22 + specular
            cr = base_r * intensity + 180 * rim_str
            cg = base_g * intensity + 190 * rim_str
            cb = base_b * intensity + 220 * rim_str
            return (
                int(max(0, min(255, cr))),
                int(max(0, min(255, cg))),
                int(max(0, min(255, cb))),
            )

        vertex_normals = {}
        valid_tris = []
        for tri in FACE_MESH_TRIANGLES:
            ta, tb, tc = int(tri[0]), int(tri[1]), int(tri[2])
            if ta not in points3d or tb not in points3d or tc not in points3d:
                continue
            ax2,ay2,az2 = points3d[ta]
            bx2,by2,bz2 = points3d[tb]
            cx2,cy2,cz2 = points3d[tc]
            ux,uy,uz = bx2-ax2, by2-ay2, bz2-az2
            vx,vy,vz = cx2-ax2, cy2-ay2, cz2-az2
            fnx = uy*vz - uz*vy
            fny = uz*vx - ux*vz
            fnz = ux*vy - uy*vx
            fn_len = (fnx*fnx + fny*fny + fnz*fnz)**0.5
            if fn_len < 1e-7:
                continue
            fnx /= fn_len; fny /= fn_len; fnz /= fn_len
            if fnz < 0:  # нормаль до камери
                fnx, fny, fnz = -fnx, -fny, -fnz
            for vi in (ta, tb, tc):
                if vi not in vertex_normals:
                    vertex_normals[vi] = [0.0, 0.0, 0.0]
                vertex_normals[vi][0] += fnx
                vertex_normals[vi][1] += fny
                vertex_normals[vi][2] += fnz
            depth = (points3d[ta][2] + points3d[tb][2] + points3d[tc][2]) / 3.0
            valid_tris.append((depth, ta, tb, tc))

        all_ly = [points3d[vi][1] for vi in vertex_normals if vi in points3d]
        all_lx = [abs(points3d[vi][0]) for vi in vertex_normals if vi in points3d]
        face_span_y = max(1.0, max(all_ly) - min(all_ly)) if all_ly else 1.0
        face_mid_y  = (max(all_ly) + min(all_ly)) * 0.5 if all_ly else 0.0
        face_span_x = max(1.0, max(all_lx)) if all_lx else 1.0

        vertex_colors = {}
        for vi, (vnx, vny, vnz) in vertex_normals.items():
            vn_len = max(1e-9, (vnx*vnx + vny*vny + vnz*vnz)**0.5)
            lit = shade_vertex(vnx/vn_len, vny/vn_len, vnz/vn_len)
            if vi in points3d:  # лоб світліший, скроні темніші
                lx, ly, _ = points3d[vi]
                forehead_t = max(0.0, min(1.0, (face_mid_y - ly) / (face_span_y * 0.5)))
                temple_t   = max(0.0, min(1.0, (abs(lx) / face_span_x - 0.55) / 0.3))
                r = int(max(0, min(255, lit[0] + forehead_t * 16 - temple_t * 18)))
                g = int(max(0, min(255, lit[1] + forehead_t * 10 - temple_t * 10)))
                b = int(max(0, min(255, lit[2] + forehead_t *  4 - temple_t *  5)))
                vertex_colors[vi] = (r, g, b)
            else:
                vertex_colors[vi] = lit

        valid_tris.sort(key=lambda t: t[0], reverse=True)  # далі — ближче

        def mix2(c1, c2):
            return ((c1[0]+c2[0])>>1, (c1[1]+c2[1])>>1, (c1[2]+c2[2])>>1)

        def avg3(c1, c2, c3):
            return ((c1[0]+c2[0]+c3[0])//3, (c1[1]+c2[1]+c3[1])//3, (c1[2]+c2[2]+c3[2])//3)

        subdiv_depth = {"low": 0, "med": 1, "high": 2}.get(quality_mode, 1)

        def draw_tri_subdivided(pts, cols, depth):
            (p1, p2, p3), (c1, c2, c3) = pts, cols
            if depth == 0:
                pygame.draw.polygon(screen, avg3(c1, c2, c3), [p1, p2, p3])
                return
            m12 = ((p1[0]+p2[0])>>1, (p1[1]+p2[1])>>1)
            m23 = ((p2[0]+p3[0])>>1, (p2[1]+p3[1])>>1)
            m31 = ((p3[0]+p1[0])>>1, (p3[1]+p1[1])>>1)
            cm12, cm23, cm31 = mix2(c1,c2), mix2(c2,c3), mix2(c3,c1)
            d = depth - 1
            draw_tri_subdivided((p1, m12, m31), (c1, cm12, cm31), d)
            draw_tri_subdivided((m12, p2, m23), (cm12, c2, cm23), d)
            draw_tri_subdivided((m31, m23, p3), (cm31, cm23, c3), d)
            draw_tri_subdivided((m12, m23, m31), (cm12, cm23, cm31), d)

        for _depth, ta, tb, tc in valid_tris:
            if ta not in projected or tb not in projected or tc not in projected:
                continue
            ca   = vertex_colors.get(ta, (base_r, base_g, base_b))
            cb_c = vertex_colors.get(tb, (base_r, base_g, base_b))
            cc   = vertex_colors.get(tc, (base_r, base_g, base_b))
            pa = (projected[ta][0], projected[ta][1])
            pb = (projected[tb][0], projected[tb][1])
            pc = (projected[tc][0], projected[tc][1])
            draw_tri_subdivided((pa, pb, pc), (ca, cb_c, cc), subdiv_depth)

    if quality_mode == "low":
        edge_step = 3
    elif quality_mode == "med":
        edge_step = 2
    else:
        edge_step = 1

    show_wireframe = mesh_style != "solid"
    if wireframe_override is not None:
        show_wireframe = bool(wireframe_override)

    if not show_wireframe:
        return

    for a, b in FACE_MESH_EDGES[::edge_step]:
        a, b = int(a), int(b)
        if a not in projected or b not in projected:
            continue
        ax, ay, az = projected[a]
        bx, by, bz = projected[b]
        depth = (az + bz) * 0.5

        shade = int(max(80, min(240, 185 - depth * 0.22)))  # глибина → яскравість ребра
        edge_color = (78, shade, 196)
        pygame.draw.line(screen, edge_color, (ax, ay), (bx, by), mesh_line_width)

def draw_camera_mesh_overlay(
    screen,
    landmarks,
    cam_x,
    cam_y,
    draw_w,
    draw_h,
    mesh_line_width=1,
    quality_mode="med",
):
    def camera_mesh_line_px(slider: int):
        # товщина 1..8; на 1 — aaline 1px
        s = max(1, min(8, int(slider)))
        if s <= 1:
            return 1, True
        px = max(1, min(4, (s + 1) // 2))
        return px, False

    projected = {}
    for idx, lm in enumerate(landmarks):
        projected[idx] = (
            int(cam_x + lm.x * draw_w),
            int(cam_y + lm.y * draw_h),
        )

    if quality_mode == "low":
        edge_step = 3
    elif quality_mode == "med":
        edge_step = 2
    else:
        edge_step = 1

    line_w, use_thin_aa = camera_mesh_line_px(mesh_line_width)
    overlay_color = (110, 235, 255)

    for a, b in FACE_MESH_EDGES[::edge_step]:
        if a not in projected or b not in projected:
            continue
        ax, ay = projected[a]
        bx, by = projected[b]
        if use_thin_aa:
            pygame.draw.aaline(screen, overlay_color, (ax, ay), (bx, by))
        else:
            pygame.draw.line(screen, overlay_color, (ax, ay), (bx, by), line_w)

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def blend_color(color_a, color_b, t):
    t = clamp(t, 0.0, 1.0)
    return tuple(
        int(color_a[i] + (color_b[i] - color_a[i]) * t)
        for i in range(3)
    )

def draw_vertical_gradient(surface, rect, top_color, bottom_color):
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return
    gradient = pygame.Surface((w, h))
    for row in range(h):
        t = row / max(1, h - 1)
        pygame.draw.line(
            gradient,
            blend_color(top_color, bottom_color, t),
            (0, row),
            (w, row),
        )
    surface.blit(gradient, (x, y))

def draw_soft_ellipse(surface, color, rect, alpha=90, steps=4, inset_step=5):
    glow = pygame.Surface((rect.width + 24, rect.height + 24), pygame.SRCALPHA)
    for step in range(steps, 0, -1):
        t = step / steps
        inner_rect = pygame.Rect(
            12 + int((steps - step) * inset_step * 0.7),
            12 + int((steps - step) * inset_step),
            max(2, rect.width - int((steps - step) * inset_step * 1.4)),
            max(2, rect.height - int((steps - step) * inset_step * 2)),
        )
        pygame.draw.ellipse(glow, (*color, int(alpha * t * 0.42)), inner_rect)
    surface.blit(glow, (rect.x - 12, rect.y - 12))

def draw_eyes_on_projected(screen, projected, p, landmarks=None):
    """Очі по вже спроєктованих точках mesh."""
    def avg_point(indices):
        xs = [projected[i][0] for i in indices if i in projected]
        ys = [projected[i][1] for i in indices if i in projected]
        if not xs:
            return (0, 0)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    # радужка: 468 ліве, 473 праве
    eye_specs = (
        (
            [33, 133, 159, 145],
            468,
            [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
            p.get("eye_l_open", 1.0),
        ),
        (
            [362, 263, 386, 374],
            473,
            [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
            p.get("eye_r_open", 1.0),
        ),
    )
    n_lm = len(landmarks) if landmarks is not None else 0

    for iris_bbox_ids, iris_lm_idx, outline_ids, open_val in eye_specs:
        cx, cy = avg_point(iris_bbox_ids)
        left_x  = projected[iris_bbox_ids[0]][0] if iris_bbox_ids[0] in projected else cx
        right_x = projected[iris_bbox_ids[1]][0] if iris_bbox_ids[1] in projected else cx
        eye_w   = max(12, int(abs(right_x - left_x) * 1.05))

        upper_ids = outline_ids[:9]
        lower_ids = outline_ids[9:]
        upper_pts = [(projected[i][0], projected[i][1]) for i in upper_ids if i in projected]
        lower_pts = [(projected[i][0], projected[i][1]) for i in lower_ids if i in projected]

        if len(upper_pts) < 2 or len(lower_pts) < 2:
            continue

        close_t = clamp((1.0 - open_val) * 1.35, 0.0, 1.0)

        ls = sorted(lower_pts, key=lambda q: q[0])
        def _lower_y(px, _ls=ls):
            if px <= _ls[0][0]:  return _ls[0][1]
            if px >= _ls[-1][0]: return _ls[-1][1]
            for i in range(len(_ls) - 1):
                x1, y1 = _ls[i]; x2, y2 = _ls[i+1]
                if x1 <= px <= x2:
                    return int(y1 + (y2 - y1) * (px - x1) / max(1, x2 - x1))
            return _ls[-1][1]

        upper_anim = [(x, int(y + (_lower_y(x) - y) * close_t)) for x, y in upper_pts]
        pygame.draw.aalines(screen, (70, 45, 38), False, upper_anim)

        if close_t >= 0.97:
            continue

        opening = upper_anim + lower_pts
        ex0 = min(pt[0] for pt in opening) - 2
        ey0 = min(pt[1] for pt in opening) - 2
        ex1 = max(pt[0] for pt in opening) + 2
        ey1 = max(pt[1] for pt in opening) + 2
        bw  = max(4, ex1 - ex0)
        bh  = max(4, ey1 - ey0)

        eye_surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
        l_opening = [(x - ex0, y - ey0) for x, y in opening]
        pygame.draw.polygon(eye_surf, (244, 242, 238, 255), l_opening)

        iris_r = max(4, int(eye_w * 0.19))
        # зсув радужки в 2D, без артефактів Z при проекції
        if landmarks is not None and iris_lm_idx < n_lm:
            sock_xs = [landmarks[i].x for i in iris_bbox_ids if i < n_lm]
            sock_ys = [landmarks[i].y for i in iris_bbox_ids if i < n_lm]
            if sock_xs:
                sock_cx = sum(sock_xs) / len(sock_xs)
                sock_cy = sum(sock_ys) / len(sock_ys)
                l_cam = landmarks[iris_bbox_ids[0]].x if iris_bbox_ids[0] < n_lm else sock_cx
                r_cam = landmarks[iris_bbox_ids[1]].x if iris_bbox_ids[1] < n_lm else sock_cx
                eye_w_cam = max(1e-6, abs(r_cam - l_cam))
                ilm = landmarks[iris_lm_idx]
                off_x = (ilm.x - sock_cx) / eye_w_cam
                off_y = (ilm.y - sock_cy) / eye_w_cam
                iris_sx = int(cx + off_x * eye_w)
                iris_sy = int(cy + off_y * eye_w)
            else:
                iris_sx, iris_sy = int(cx), int(cy)
        else:
            gaze_x = (p.get("eye_l_gaze_x", 0) + p.get("eye_r_gaze_x", 0)) / 2
            gaze_y = (p.get("eye_l_gaze_y", 0) + p.get("eye_r_gaze_y", 0)) / 2
            iris_sx = int(cx + gaze_x * eye_w * 0.25)
            iris_sy = int(cy + gaze_y * eye_w * 0.25)
        ilx = clamp(iris_sx - ex0, iris_r, bw - iris_r)
        ily = clamp(iris_sy - ey0, iris_r, bh - iris_r)
        pygame.draw.circle(eye_surf, (28,  40,  58, 255), (ilx, ily), iris_r)
        pygame.draw.circle(eye_surf, (54,  78, 104, 255), (ilx, ily), max(3, iris_r - 1))
        pygame.draw.circle(eye_surf, (80, 115, 155, 255), (ilx, ily), max(2, int(iris_r * 0.72)))
        pygame.draw.circle(eye_surf, (44,  64,  90, 255), (ilx, ily), max(2, int(iris_r * 0.55)))
        pygame.draw.circle(eye_surf, (14,  14,  18, 255), (ilx, ily), max(1, int(iris_r * 0.40)))
        if iris_r >= 4:
            pygame.draw.circle(eye_surf, (255, 255, 255, 255), (ilx - iris_r // 3, ily - iris_r // 3), max(1, iris_r // 4))
        if iris_r >= 6:
            pygame.draw.circle(eye_surf, (180, 210, 255, 120), (ilx + iris_r // 4, ily + iris_r // 4), max(1, iris_r // 5))

        mask = pygame.Surface((bw, bh), pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), l_opening)
        eye_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(eye_surf, (ex0, ey0))
        pygame.draw.aalines(screen, (160, 130, 118), False, lower_pts)


_avatar_bg_cache: dict = {}

AVATAR_SCALE = 0.82

def draw_avatar(screen, p, off_x, off_y, landmarks, yaw, pitch, roll, center_x, center_y, cam_aspect, quality_mode="med"):
    panel_rect = pygame.Rect(RENDER_PANEL_X, MENU_H, RENDER_PANEL_W, HEIGHT - MENU_H)

    cache_key = (panel_rect.width, panel_rect.height)
    if _avatar_bg_cache.get("key") != cache_key:
        bg = pygame.Surface((panel_rect.width, panel_rect.height))
        draw_vertical_gradient(bg, (0, 0, panel_rect.width, panel_rect.height), (28, 39, 58), (17, 20, 29))
        halo = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        pygame.draw.ellipse(halo, (84, 129, 204, 42), (panel_rect.width // 2 - 180, 34, 360, 220))
        pygame.draw.ellipse(halo, (244, 160, 128, 28), (panel_rect.width // 2 - 130, panel_rect.height - 210, 260, 140))
        _avatar_bg_cache["key"] = cache_key
        _avatar_bg_cache["bg"]  = bg
        _avatar_bg_cache["halo"] = halo
    screen.blit(_avatar_bg_cache["bg"],   panel_rect.topleft)
    screen.blit(_avatar_bg_cache["halo"], panel_rect.topleft)

    use_ss = (quality_mode == "high")  # подвійна роздільність лише на high
    SS = 2 if use_ss else 1
    ss_w = max(1, panel_rect.width * SS)
    ss_h = max(1, panel_rect.height * SS)
    ss_surf = pygame.Surface((ss_w, ss_h))
    ss_surf.fill((20, 28, 42))
    ss_center_x = (center_x - RENDER_PANEL_X) * SS
    ss_center_y = (center_y - MENU_H) * SS

    draw_3d_face_mesh(
        ss_surf,
        landmarks,
        off_x,
        off_y,
        ss_center_x,
        ss_center_y,
        cam_aspect,
        AVATAR_SCALE,
        1,
        "solid",
        quality_mode,
        face_fill_color=(210, 178, 158),
        wireframe_override=False,
        panel_w=ss_w,
        panel_h=ss_h,
    )

    if use_ss:
        downscaled = pygame.transform.smoothscale(ss_surf, (panel_rect.width, panel_rect.height))
        screen.blit(downscaled, panel_rect.topleft)
    else:
        screen.blit(ss_surf, panel_rect.topleft)

    projected, _points3d = project_face_landmarks(
        landmarks, off_x, off_y, center_x, center_y, cam_aspect, AVATAR_SCALE,
        panel_h=panel_rect.height
    )
    if not projected:
        return

    def avg_point(indices):
        xs = [projected[i][0] for i in indices if i in projected]
        ys = [projected[i][1] for i in indices if i in projected]
        if not xs:
            return (0, 0)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    draw_eyes_on_projected(screen, projected, p, landmarks)

    if 234 in projected and 454 in projected:
        _brow_face_w = max(1, abs(projected[454][0] - projected[234][0]))
    else:
        _brow_face_w = 200
    brow_up_max = _brow_face_w * 0.06
    brow_dn_max = _brow_face_w * 0.006
    brow_shift  = int(_brow_face_w * 0.018)

    brow_specs = [
        [107, 66, 105, 63, 70],
        [300, 293, 334, 296, 336],
    ]
    for brow_ids in brow_specs:
        base = [(projected[i][0], projected[i][1]) for i in brow_ids if i in projected]
        if len(base) < 3:
            continue
        n = len(base)
        upper, lower = [], []
        for i, (x, y) in enumerate(base):
            t = i / max(1, n - 1)
            if t <= 0.42:
                arch = max(0.40, t / 0.42)
            else:
                arch = max(0.14, 1.0 - (t - 0.42) / 0.58 * 0.86)
            upper.append((x, y + brow_shift - int(brow_up_max * arch)))
            lower.append((x, y + brow_shift + int(brow_dn_max * arch)))
        bx0 = min(x for x, y in upper + lower) - 4
        by0 = min(y for x, y in upper + lower) - 4
        bx1 = max(x for x, y in upper + lower) + 4
        by1 = max(y for x, y in upper + lower) + 4
        bsw  = max(4, bx1 - bx0)
        bsh  = max(4, by1 - by0)
        bsurf = pygame.Surface((bsw, bsh), pygame.SRCALPHA)
        lu = [(x - bx0, y - by0) for x, y in upper]
        ll = [(x - bx0, y - by0) for x, y in lower]
        su = [(x, y - 2) for x, y in lu]
        sl = [(x, y + 2) for x, y in ll]
        pygame.draw.polygon(bsurf, (108, 82, 58, 100), su + list(reversed(sl)))
        pygame.draw.polygon(bsurf, (60, 42, 26, 235), lu + list(reversed(ll)))
        pygame.draw.aalines(bsurf, (32, 22, 12, 255), False, lu)
        screen.blit(bsurf, (bx0, by0))

    if 114 in projected and 188 in projected and 245 in projected:
        nose_l = [(projected[i][0], projected[i][1]) for i in [193, 122, 188] if i in projected]
        if len(nose_l) >= 2:
            pygame.draw.aalines(screen, (172, 138, 122), False, nose_l)
    if 343 in projected and 412 in projected and 465 in projected:
        nose_r = [(projected[i][0], projected[i][1]) for i in [417, 351, 399] if i in projected]
        if len(nose_r) >= 2:
            pygame.draw.aalines(screen, (172, 138, 122), False, nose_r)

    left_nostril = avg_point([98, 97, 2])
    right_nostril = avg_point([326, 327, 2])
    nostril_w = 5
    nostril_h = 3
    pygame.draw.arc(
        screen,
        (130, 104, 92),
        pygame.Rect(int(left_nostril[0] - nostril_w), int(left_nostril[1] - 1), nostril_w * 2, nostril_h * 2),
        3.5,
        6.0,
        1,
    )
    pygame.draw.arc(
        screen,
        (130, 104, 92),
        pygame.Rect(int(right_nostril[0] - nostril_w), int(right_nostril[1] - 1), nostril_w * 2, nostril_h * 2),
        3.4,
        5.9,
        1,
    )

    outer_upper_ids = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    outer_lower_ids = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    inner_upper_ids = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    inner_lower_ids = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

    def lip_pts(ids):
        return [(projected[i][0], projected[i][1]) for i in ids if i in projected]

    outer_upper = lip_pts(outer_upper_ids)
    outer_lower = lip_pts(outer_lower_ids)
    inner_upper = lip_pts(inner_upper_ids)
    inner_lower = lip_pts(inner_lower_ids)

    if len(outer_upper) >= 3 and len(outer_lower) >= 3:
        def _thin(outer, inner, t=0.55):
            n = min(len(outer), len(inner))
            return [(int(outer[i][0] + (inner[i][0] - outer[i][0]) * t),
                     int(outer[i][1] + (inner[i][1] - outer[i][1]) * t))
                    for i in range(n)]

        thin_upper = _thin(outer_upper, inner_upper, 0.55) if len(inner_upper) >= 3 else outer_upper
        thin_lower = _thin(outer_lower, inner_lower, 0.50) if len(inner_lower) >= 3 else outer_lower

        if len(thin_lower) >= 3:
            pygame.draw.polygon(screen, (188, 130, 130), thin_lower)

        pygame.draw.polygon(screen, (162, 100, 108), thin_upper)

        if thin_upper and thin_lower:
            pygame.draw.aaline(screen, (148, 95, 102), thin_upper[0],  thin_lower[0])
            pygame.draw.aaline(screen, (148, 95, 102), thin_upper[-1], thin_lower[-1])

        lower_mid     = outer_lower[len(outer_lower) // 2]
        lower_left_q  = outer_lower[len(outer_lower) // 4]
        lower_right_q = outer_lower[3 * len(outer_lower) // 4]
        highlight_pts = [lower_left_q, lower_mid, lower_right_q]
        hl_xs = [pt[0] for pt in highlight_pts]
        hl_ys = [pt[1] for pt in highlight_pts]
        hl_x0 = min(hl_xs) - 4;  hl_y0 = min(hl_ys) - 4
        hl_w  = max(4, max(hl_xs) - hl_x0 + 4)
        hl_h  = max(4, max(hl_ys) - hl_y0 + 4)
        hl_surf = pygame.Surface((hl_w, hl_h), pygame.SRCALPHA)
        pygame.draw.lines(hl_surf, (220, 175, 170, 90), False,
                          [(x - hl_x0, y - hl_y0) for x, y in highlight_pts], 3)
        screen.blit(hl_surf, (hl_x0, hl_y0))

        if len(inner_upper) >= 3 and len(inner_lower) >= 3:
            mouth_open_poly = inner_upper + list(reversed(inner_lower[1:-1]))
            if len(mouth_open_poly) >= 3:
                mi_x0 = min(pt[0] for pt in mouth_open_poly) - 1
                mi_y0 = min(pt[1] for pt in mouth_open_poly) - 1
                mi_x1 = max(pt[0] for pt in mouth_open_poly) + 1
                mi_y1 = max(pt[1] for pt in mouth_open_poly) + 1
                mi_w  = max(4, mi_x1 - mi_x0)
                mi_h  = max(4, mi_y1 - mi_y0)
                mouth_int_surf = pygame.Surface((mi_w, mi_h), pygame.SRCALPHA)
                m_local = [(x - mi_x0, y - mi_y0) for x, y in mouth_open_poly]
                pygame.draw.polygon(mouth_int_surf, (50, 22, 28, 255), m_local)

                jaw_open = p.get("mouth_open", 0.0)
                if jaw_open > 0.03:
                    iu_loc = [(pt[0] - mi_x0, pt[1] - mi_y0) for pt in inner_upper]
                    il_loc = [(pt[0] - mi_x0, pt[1] - mi_y0) for pt in inner_lower]

                    if 234 in projected and 454 in projected:
                        face_w_px = max(1, abs(projected[454][0] - projected[234][0]))
                    else:
                        face_w_px = max(1, abs(inner_upper[-1][0] - inner_upper[0][0])) * 2
                    t_half_w = face_w_px * 0.22
                    mouth_cx = (mi_x1 - mi_x0) / 2.0

                    def clamp_x(pts):
                        return [(clamp(x, mouth_cx - t_half_w, mouth_cx + t_half_w), y)
                                for x, y in pts]

                    iu_loc = clamp_x(iu_loc)
                    il_loc = clamp_x(il_loc)

                    iu_avg_y = sum(y for _, y in iu_loc) / len(iu_loc)
                    il_avg_y = sum(y for _, y in il_loc) / len(il_loc)
                    mouth_h  = max(1, il_avg_y - iu_avg_y)

                    t_up_h  = max(2, int(mouth_h * 0.44 * min(1.0, jaw_open * 3.5)))
                    t_up_lo = [(x, y + t_up_h) for x, y in iu_loc]
                    pygame.draw.polygon(mouth_int_surf, (242, 238, 230, 255),
                                        iu_loc + list(reversed(t_up_lo)))
                    pygame.draw.aalines(mouth_int_surf, (190, 118, 118, 180), False, iu_loc)
                    if jaw_open > 0.12:
                        t_dn_h  = max(2, int(mouth_h * 0.28 * min(1.0, jaw_open * 2.5)))
                        t_dn_hi = [(x, y - t_dn_h) for x, y in il_loc]
                        pygame.draw.polygon(mouth_int_surf, (232, 228, 220, 255),
                                            t_dn_hi + list(reversed(il_loc)))
                        pygame.draw.aalines(mouth_int_surf, (190, 118, 118, 150), False, il_loc)

                mi_clip = pygame.Surface((mi_w, mi_h), pygame.SRCALPHA)
                pygame.draw.polygon(mi_clip, (255, 255, 255, 255), m_local)
                mouth_int_surf.blit(mi_clip, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(mouth_int_surf, (mi_x0, mi_y0))

            tongue_out_val = p.get("tongue_out", 0.0)
            mouth_open_val = p.get("mouth_open", 0.0)
            if tongue_out_val > 0.08 and mouth_open_val > 0.08 and len(mouth_open_poly) >= 3:
                mx0 = min(pt[0] for pt in mouth_open_poly) - 2
                my0 = min(pt[1] for pt in mouth_open_poly) - 2
                mx1 = max(pt[0] for pt in mouth_open_poly) + 2
                my1 = max(pt[1] for pt in mouth_open_poly) + 2
                mw  = max(4, mx1 - mx0)
                mh  = max(4, my1 - my0)
                t_cx     = sum(pt[0] for pt in inner_lower) / len(inner_lower)
                t_cy_low = sum(pt[1] for pt in inner_lower) / len(inner_lower)
                t_cy_hi  = sum(pt[1] for pt in inner_upper) / len(inner_upper)
                t_half_w = max(4, int((inner_lower[-1][0] - inner_lower[0][0]) * 0.35))
                t_half_h = max(3, int((t_cy_low - t_cy_hi) * 0.5 *
                                      min(1.0, tongue_out_val * 2.0 + mouth_open_val * 0.3)))
                t_cy = int(t_cy_low - t_half_h * 0.5)
                tongue_surf = pygame.Surface((mw, mh), pygame.SRCALPHA)
                pygame.draw.ellipse(tongue_surf, (200, 90, 105, 240),
                    (int(t_cx - mx0) - t_half_w, int(t_cy - my0) - t_half_h,
                     t_half_w * 2, t_half_h * 2))
                pygame.draw.ellipse(tongue_surf, (225, 140, 150, 90),
                    (int(t_cx - mx0) - t_half_w // 2, int(t_cy - my0) - int(t_half_h * 0.7),
                     t_half_w, int(t_half_h * 0.8)))
                if t_half_w >= 6:
                    pygame.draw.line(tongue_surf, (160, 65, 78, 120),
                        (int(t_cx - mx0), int(t_cy - my0) - t_half_h + 2),
                        (int(t_cx - mx0), int(t_cy - my0) + t_half_h - 2), 1)
                mask_t = pygame.Surface((mw, mh), pygame.SRCALPHA)
                local_mouth_pts = [(x - mx0, y - my0) for x, y in mouth_open_poly]
                pygame.draw.polygon(mask_t, (255, 255, 255, 255), local_mouth_pts)
                tongue_surf.blit(mask_t, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(tongue_surf, (mx0, my0))

        pygame.draw.aalines(screen, (130, 78, 84), False, thin_upper)

        if 0 in projected and 267 in projected and 37 in projected:
            ph_top = projected[0]
            ph_l = projected[37]
            ph_r = projected[267]
            pygame.draw.aaline(screen, (170, 130, 118), (ph_l[0], ph_l[1]), (ph_top[0], ph_top[1] - 3))
            pygame.draw.aaline(screen, (170, 130, 118), (ph_r[0], ph_r[1]), (ph_top[0], ph_top[1] - 3))
    
def handle_events(events, loop_state):
    """Обробка подій; True — вихід з програми."""
    rs  = loop_state["runtime_settings"]
    for event in events:
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.VIDEORESIZE:
            global WIDTH, HEIGHT
            WIDTH  = max(860, event.w)
            HEIGHT = max(460, event.h)
            loop_state["WIDTH"]  = WIDTH
            loop_state["HEIGHT"] = HEIGHT
            loop_state["screen"] = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            recalc_layout()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mb = loop_state["menu_buttons"]
            now = pygame.time.get_ticks
            changed_menu = True
            if mb.get("mode_avatar") and mb["mode_avatar"].collidepoint(event.pos):
                loop_state["render_mode"] = "avatar"
            elif mb.get("mode_mesh3d") and mb["mode_mesh3d"].collidepoint(event.pos):
                loop_state["render_mode"] = "mesh3d"
            elif mb.get("mode_cam") and mb["mode_cam"].collidepoint(event.pos):
                loop_state["render_mode"] = "camera_mesh"
                loop_state["toast"] = ("Mode: Camera Mesh", now() + 1300)
            elif mb.get("layout") and mb["layout"].collidepoint(event.pos):
                idx = LAYOUT_PRESETS.index(rs["layoutPreset"])
                rs["layoutPreset"] = LAYOUT_PRESETS[(idx + 1) % len(LAYOUT_PRESETS)]
                recalc_layout()
                loop_state["toast"] = (
                    f"Layout: {LAYOUT_LABELS.get(rs['layoutPreset'], rs['layoutPreset'])}",
                    now() + 1200)
            elif mb.get("camera_toggle") and mb["camera_toggle"].collidepoint(event.pos):
                rs["showCameraPanel"] = not rs["showCameraPanel"]
                recalc_layout()
            elif mb.get("quality") and mb["quality"].collidepoint(event.pos):
                idx = QUALITY_MODES.index(rs["qualityMode"])
                rs["qualityMode"] = QUALITY_MODES[(idx + 1) % len(QUALITY_MODES)]
                loop_state["toast"] = (f"Quality: {rs['qualityMode']}", now() + 1300)
            elif mb.get("fps") and mb["fps"].collidepoint(event.pos):
                idx = FPS_LIMITS.index(rs["fpsLimit"])
                rs["fpsLimit"] = FPS_LIMITS[(idx + 1) % len(FPS_LIMITS)]
                label = "Unlimited" if rs["fpsLimit"] == 0 else str(rs["fpsLimit"])
                loop_state["toast"] = (f"FPS limit: {label}", now() + 1300)
            elif mb.get("scale_minus") and mb["scale_minus"].collidepoint(event.pos):
                rs["meshScale"] = max(0.6, rs["meshScale"] - 0.05)
            elif mb.get("scale_plus") and mb["scale_plus"].collidepoint(event.pos):
                rs["meshScale"] = min(2.2, rs["meshScale"] + 0.05)
            elif mb.get("style") and mb["style"].collidepoint(event.pos):
                idx = MESH_STYLES.index(rs["meshStyle"])
                rs["meshStyle"] = MESH_STYLES[(idx + 1) % len(MESH_STYLES)]
                loop_state["toast"] = (f"Mesh style: {rs['meshStyle']}", now() + 1300)
            elif mb.get("line_minus") and mb["line_minus"].collidepoint(event.pos):
                if loop_state["render_mode"] == "camera_mesh":
                    rs["cameraMeshLineWidth"] = max(1, rs["cameraMeshLineWidth"] - 1)
                    loop_state["toast"] = (f"Line: {rs['cameraMeshLineWidth']}", now() + 1000)
                else:
                    rs["meshLineWidth"] = max(1, rs["meshLineWidth"] - 1)
                    loop_state["toast"] = (f"Line: {rs['meshLineWidth']}", now() + 1000)
            elif mb.get("line_plus") and mb["line_plus"].collidepoint(event.pos):
                if loop_state["render_mode"] == "camera_mesh":
                    rs["cameraMeshLineWidth"] = min(8, rs["cameraMeshLineWidth"] + 1)
                    loop_state["toast"] = (f"Line: {rs['cameraMeshLineWidth']}", now() + 1000)
                else:
                    rs["meshLineWidth"] = min(5, rs["meshLineWidth"] + 1)
                    loop_state["toast"] = (f"Line: {rs['meshLineWidth']}", now() + 1000)
            elif mb.get("hud_toggle") and mb["hud_toggle"].collidepoint(event.pos):
                rs["showHud"] = not rs["showHud"]
            elif mb.get("baseline") and mb["baseline"].collidepoint(event.pos):
                if loop_state["params"]:
                    pr = loop_state["params"]
                    loop_state["neutral_baseline"] = {
                        key: pr.get(key, 0.0)
                        for key in ("mouth_smile", "mouth_frown", "mouth_open",
                                    "mouth_pucker", "squint", "brow_l_up", "brow_r_up")
                    }
                    loop_state["toast"] = ("Neutral baseline captured", now() + 1500)
            elif mb.get("reset") and mb["reset"].collidepoint(event.pos):
                loop_state["runtime_settings"] = rs = normalize_settings(DEFAULT_SETTINGS)
                recalc_layout()
                loop_state["neutral_baseline"] = {}
                loop_state["toast"] = ("Settings reset to defaults", now() + 1500)
            else:
                changed_menu = False
            if changed_menu:
                save_settings(rs)
        if event.type == pygame.KEYDOWN:
            changed = True
            k = event.key
            if k == pygame.K_1:
                loop_state["render_mode"] = "avatar";  changed = False
            elif k == pygame.K_2:
                loop_state["render_mode"] = "mesh3d";  changed = False
            elif k == pygame.K_3:
                loop_state["render_mode"] = "camera_mesh"
                loop_state["toast"] = ("Mode: Camera Mesh", pygame.time.get_ticks() + 1300)
                changed = False
            elif k == pygame.K_h:
                rs["showHud"] = not rs["showHud"]
            elif k == pygame.K_c:
                rs["showCameraPanel"] = not rs["showCameraPanel"];  recalc_layout()
            elif k == pygame.K_TAB:
                idx = LAYOUT_PRESETS.index(rs["layoutPreset"])
                rs["layoutPreset"] = LAYOUT_PRESETS[(idx + 1) % len(LAYOUT_PRESETS)]
                recalc_layout()
                loop_state["toast"] = (
                    f"Layout: {LAYOUT_LABELS.get(rs['layoutPreset'], rs['layoutPreset'])}",
                    pygame.time.get_ticks() + 1200)
            elif k in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                rs["meshScale"] = min(2.2, rs["meshScale"] + 0.05)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                rs["meshScale"] = max(0.6, rs["meshScale"] - 0.05)
            elif k == pygame.K_LEFTBRACKET:
                if loop_state["render_mode"] == "camera_mesh":
                    rs["cameraMeshLineWidth"] = max(1, rs["cameraMeshLineWidth"] - 1)
                    loop_state["toast"] = (f"Camera mesh line: {rs['cameraMeshLineWidth']}",
                                           pygame.time.get_ticks() + 1200)
                else:
                    rs["meshLineWidth"] = max(1, rs["meshLineWidth"] - 1)
            elif k == pygame.K_RIGHTBRACKET:
                if loop_state["render_mode"] == "camera_mesh":
                    rs["cameraMeshLineWidth"] = min(8, rs["cameraMeshLineWidth"] + 1)
                    loop_state["toast"] = (f"Camera mesh line: {rs['cameraMeshLineWidth']}",
                                           pygame.time.get_ticks() + 1200)
                else:
                    rs["meshLineWidth"] = min(5, rs["meshLineWidth"] + 1)
            elif k == pygame.K_v:
                idx = MESH_STYLES.index(rs["meshStyle"])
                rs["meshStyle"] = MESH_STYLES[(idx + 1) % len(MESH_STYLES)]
                loop_state["toast"] = (f"Mesh style: {rs['meshStyle']}",
                                       pygame.time.get_ticks() + 1300)
            elif k == pygame.K_q:
                idx = QUALITY_MODES.index(rs["qualityMode"])
                rs["qualityMode"] = QUALITY_MODES[(idx + 1) % len(QUALITY_MODES)]
                loop_state["toast"] = (f"Quality: {rs['qualityMode']}",
                                       pygame.time.get_ticks() + 1300)
            elif k == pygame.K_f:
                idx = FPS_LIMITS.index(rs["fpsLimit"])
                rs["fpsLimit"] = FPS_LIMITS[(idx + 1) % len(FPS_LIMITS)]
                label = "Unlimited" if rs["fpsLimit"] == 0 else str(rs["fpsLimit"])
                loop_state["toast"] = (f"FPS limit: {label}", pygame.time.get_ticks() + 1300)
            elif k == pygame.K_k and loop_state["params"]:
                pr = loop_state["params"]
                loop_state["neutral_baseline"] = {
                    key: pr.get(key, 0.0)
                    for key in ("mouth_smile", "mouth_frown", "mouth_open",
                                "mouth_pucker", "squint", "brow_l_up", "brow_r_up")
                }
                loop_state["toast"] = ("Neutral baseline captured", pygame.time.get_ticks() + 1500)
                changed = False
            elif k == pygame.K_r:
                loop_state["runtime_settings"] = rs = normalize_settings(DEFAULT_SETTINGS)
                recalc_layout()
                loop_state["neutral_baseline"] = {}
                loop_state["toast"] = ("Settings reset to defaults", pygame.time.get_ticks() + 1500)
            else:
                changed = False
            if changed:
                save_settings(rs)
    return False


def draw_menu_bar(screen, loop_state):
    """Верхнє меню; словник rect-ів кнопок."""
    rs = loop_state["runtime_settings"]
    rm = loop_state["render_mode"]
    W  = loop_state["WIDTH"]

    bar_surf = pygame.Surface((W, MENU_H))
    bar_surf.fill((18, 22, 32))
    screen.blit(bar_surf, (0, 0))
    pygame.draw.line(screen, (50, 60, 80), (0, MENU_H - 1), (W, MENU_H - 1))

    font = pygame.font.SysFont("Menlo", 13)
    btn_h = MENU_H - 10
    btn_y = 5
    pad_x = 11
    gap   = 5
    sep   = 16

    buttons = {}

    def _btn(label, x, active=False, accent=False):
        tw   = font.size(label)[0]
        bw   = tw + pad_x * 2
        rect = pygame.Rect(x, btn_y, bw, btn_h)
        if active:
            bg = (40, 90, 160)
            border = (80, 160, 240)
        elif accent:
            bg = (50, 38, 20)
            border = (180, 130, 60)
        else:
            bg = (28, 34, 48)
            border = (55, 65, 85)
        pygame.draw.rect(screen, bg, rect, border_radius=5)
        pygame.draw.rect(screen, border, rect, 1, border_radius=5)
        txt = font.render(label, True, (230, 240, 255) if active else (190, 200, 215))
        screen.blit(txt, txt.get_rect(center=rect.center))
        return rect, x + bw + gap

    x = 8

    r, x = _btn("Avatar",  x, active=(rm == "avatar"))
    buttons["mode_avatar"] = r
    r, x = _btn("Mesh 3D", x, active=(rm == "mesh3d"))
    buttons["mode_mesh3d"] = r
    r, x = _btn("Cam",     x, active=(rm == "camera_mesh"))
    buttons["mode_cam"] = r

    x += sep

    preset = rs["layoutPreset"]
    layout_short = {"50_50": "50/50", "40_60": "40/60", "30_70": "30/70",
                    "camera_only": "Cam only", "render_only": "Rndr only"}
    r, x = _btn(f"Layout: {layout_short.get(preset, preset)}", x)
    buttons["layout"] = r

    r, x = _btn(f"Camera: {'ON' if rs['showCameraPanel'] else 'OFF'}", x,
                active=rs["showCameraPanel"])
    buttons["camera_toggle"] = r

    x += sep

    r, x = _btn(f"Q: {rs['qualityMode'].upper()}", x)
    buttons["quality"] = r

    fps_label = "FPS: ∞" if rs["fpsLimit"] == 0 else f"FPS: {rs['fpsLimit']}"
    r, x = _btn(fps_label, x)
    buttons["fps"] = r

    x += sep

    r, x = _btn("Scale-", x)
    buttons["scale_minus"] = r
    r, x = _btn(f"{rs['meshScale']:.2f}", x)
    buttons["scale_val"] = r
    r, x = _btn("Scale+", x)
    buttons["scale_plus"] = r

    x += sep

    style_icons = {"wire": "Wire", "hybrid": "Hybrid", "solid": "Solid"}
    r, x = _btn(style_icons.get(rs["meshStyle"], rs["meshStyle"]), x)
    buttons["style"] = r

    x += sep

    lw = rs["cameraMeshLineWidth"] if rm == "camera_mesh" else rs["meshLineWidth"]
    r, x = _btn("Line-", x)
    buttons["line_minus"] = r
    r, x = _btn(f"W:{lw}", x)
    buttons["line_val"] = r
    r, x = _btn("Line+", x)
    buttons["line_plus"] = r

    x += sep

    r, x = _btn("HUD", x, active=rs["showHud"])
    buttons["hud_toggle"] = r

    reset_label   = "Reset"
    base_label    = "Baseline"
    rw_reset  = font.size(reset_label)[0]  + pad_x * 2
    rw_base   = font.size(base_label)[0]   + pad_x * 2
    rx_reset  = W - rw_reset - 8
    rx_base   = rx_reset - rw_base - gap
    r2, _ = _btn(reset_label, rx_reset)
    buttons["reset"] = r2
    r2, _ = _btn(base_label, rx_base, accent=True)
    buttons["baseline"] = r2

    return buttons


def render_scene(screen, rgb, loop_state, cam_aspect, smoothed_x, smoothed_y,
                 smoothed_yaw, smoothed_pitch, smoothed_roll,
                 current_face_landmarks, has_face, clock_fps):
    """Малює кадр; повертає rect-и меню."""
    rs            = loop_state["runtime_settings"]
    render_mode   = loop_state["render_mode"]
    params        = loop_state["params"]
    emotion_state = loop_state["emotion_state"]
    layout_preset = rs["layoutPreset"]
    cam_h, cam_w  = rgb.shape[:2]
    cam_x = cam_y = draw_w = draw_h = 0

    screen.fill((20, 22, 26))

    def _blit_camera(fit_w, fit_h, off_x=0, off_y=0):
        nonlocal cam_x, cam_y, draw_w, draw_h
        fs     = min(fit_w / cam_w, fit_h / cam_h)
        draw_w = max(1, int(cam_w * fs))
        draw_h = max(1, int(cam_h * fs))
        preview    = cv2.resize(rgb, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
        cam_surf   = pygame.image.frombuffer(preview.tobytes(), (draw_w, draw_h), "RGB")
        cam_x = off_x + (fit_w - draw_w) // 2
        avail_h = loop_state["HEIGHT"] - MENU_H
        cam_y = MENU_H + max(0, (avail_h - draw_h) // 2)
        screen.blit(cam_surf, (cam_x, cam_y))

    if render_mode == "camera_mesh":
        _blit_camera(loop_state["WIDTH"], loop_state["HEIGHT"])
    elif layout_preset == "camera_only":
        _blit_camera(loop_state["WIDTH"], loop_state["HEIGHT"])
    elif layout_preset != "render_only" and rs["showCameraPanel"]:
        _blit_camera(CAMERA_PANEL_W, loop_state["HEIGHT"])
        pygame.draw.rect(screen, (36, 38, 44), (CAMERA_PANEL_W, 0, GAP, loop_state["HEIGHT"]))

    if render_mode != "camera_mesh":
        pygame.draw.rect(screen, (28, 30, 36), (RENDER_PANEL_X, 0, RENDER_PANEL_W, loop_state["HEIGHT"]))

    if current_face_landmarks is not None:
        if render_mode == "mesh3d" and layout_preset != "camera_only":
            draw_3d_face_mesh(screen, current_face_landmarks, smoothed_x, smoothed_y,
                              RENDER_CENTER_X, RENDER_CENTER_Y, cam_aspect,
                              rs["meshScale"], rs["meshLineWidth"],
                              rs["meshStyle"], rs["qualityMode"])
            if params:
                _proj, _ = project_face_landmarks(
                    current_face_landmarks, smoothed_x, smoothed_y,
                    RENDER_CENTER_X, RENDER_CENTER_Y, cam_aspect, rs["meshScale"])
                if _proj:
                    draw_eyes_on_projected(screen, _proj, params, current_face_landmarks)

        elif render_mode == "camera_mesh":
            if draw_w > 0 and draw_h > 0:
                draw_camera_mesh_overlay(screen, current_face_landmarks,
                                         cam_x, cam_y, draw_w, draw_h,
                                         rs["cameraMeshLineWidth"], rs["qualityMode"])
                n_lm = len(current_face_landmarks)
                for iris_lm_i in (468, 473):
                    if iris_lm_i < n_lm:
                        lm  = current_face_landmarks[iris_lm_i]
                        icx = int(cam_x + lm.x * draw_w)
                        icy = int(cam_y + lm.y * draw_h)
                        pygame.draw.circle(screen, (80, 210, 255), (icx, icy), 5, 1)
                        pygame.draw.circle(screen, (20, 20, 20),   (icx, icy), 2)

        elif params and layout_preset != "camera_only":
            draw_avatar(screen, params, smoothed_x, smoothed_y, current_face_landmarks,
                        smoothed_yaw, smoothed_pitch, smoothed_roll,
                        RENDER_CENTER_X, RENDER_CENTER_Y, cam_aspect, rs["qualityMode"])
    else:
        font = pygame.font.SysFont(None, 32)
        msg  = font.render("Face not detected", True, (180, 180, 190))
        if render_mode == "camera_mesh" or layout_preset == "camera_only":
            msg_rect = msg.get_rect(center=(loop_state["WIDTH"] // 2, loop_state["HEIGHT"] // 2))
        else:
            msg_rect = msg.get_rect(center=(RENDER_CENTER_X, RENDER_CENTER_Y))
        screen.blit(msg, msg_rect)

    if rs["showHud"]:
        draw_hud(screen, params, smoothed_yaw, smoothed_pitch, smoothed_roll,
                 has_face, clock_fps, emotion_state)

    menu_buttons = draw_menu_bar(screen, loop_state)

    toast_msg, toast_until = loop_state["toast"]
    draw_toast(screen, toast_msg, toast_until)
    pygame.display.flip()
    return menu_buttons


params = None
smoothed_x, smoothed_y = 0.0, 0.0
smoothed_yaw, smoothed_pitch, smoothed_roll = 0.0, 0.0, 0.0
render_mode = "mesh3d"
emotion_state = {
    "smooth_scores": {
        "Happy": 0.0,
        "Sad": 0.0,
        "Surprised": 0.0,
        "Excited": 0.0,
        "Angry": 0.0,
        "Focused": 0.0,
        "Disgusted": 0.0,
        "Thinking": 0.0,
        "Duck Face": 0.0,
        "Neutral": 0.08,
    },
    "active_label": "Neutral",
    "active_score": 0.08,
    "top3": [],
}
neutral_baseline = {}
toast_message = ""
toast_until_ms = 0

CONTOUR_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 
    234, 127, 162, 21, 54, 103, 67, 109
]

loop_state = {
    "runtime_settings": runtime_settings,
    "render_mode":      "mesh3d",
    "params":           None,
    "emotion_state":    emotion_state,
    "neutral_baseline": {},
    "menu_buttons":     {},
    "toast":            ("", 0),
    "WIDTH":            WIDTH,
    "HEIGHT":           HEIGHT,
    "screen":           screen,
}

try:
    current_face_landmarks = None
    _frame_counter = 0
    _last_detection_result = None
    smoothed_x = smoothed_y = 0.0
    smoothed_yaw = smoothed_pitch = smoothed_roll = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rs = loop_state["runtime_settings"]
        detect_target_w = DETECT_WIDTH_BY_QUALITY.get(rs["qualityMode"], 480)
        detect_rgb = rgb
        if detect_rgb.shape[1] > detect_target_w:
            dh = max(1, int(detect_rgb.shape[0] * detect_target_w / detect_rgb.shape[1]))
            detect_rgb = cv2.resize(detect_rgb, (detect_target_w, dh), interpolation=cv2.INTER_LINEAR)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=detect_rgb)

        _frame_counter += 1
        _stride     = max(1, int(DETECT_STRIDE_BY_QUALITY.get(rs["qualityMode"], 2)))
        _run_detect = _last_detection_result is None or (_frame_counter % _stride == 0)
        if _run_detect:
            result = detector.detect(mp_image)
            _last_detection_result = result
        else:
            result = _last_detection_result

        has_face = bool(result.face_landmarks)
        current_face_landmarks = result.face_landmarks[0] if has_face else None

        if has_face and result.face_blendshapes and _run_detect:
            new_params = get_face_params(result.face_blendshapes[0])
            if loop_state["params"] is None:
                loop_state["params"] = new_params
            else:
                p = loop_state["params"]
                for k in p:
                    diff = abs(p[k] - new_params[k])
                    if diff < 0.01:
                        a = 0.02
                    elif "eye" in k and diff > 0.1:
                        a = 0.8
                    elif "squint" in k:
                        a = 0.1
                    else:
                        a = 0.2
                    p[k] = p[k] * (1 - a) + new_params[k] * a
            loop_state["emotion_state"] = update_emotion_state(
                loop_state["params"], loop_state["emotion_state"], loop_state["neutral_baseline"])

        if has_face and current_face_landmarks is not None:
            nose   = current_face_landmarks[1]
            face_x = 0.5 - nose.x
            face_y = nose.y - 0.5
            smoothed_x = 0.15 * face_x + 0.85 * smoothed_x
            smoothed_y = 0.15 * face_y + 0.85 * smoothed_y
            raw_yaw, raw_pitch, raw_roll = estimate_head_pose(current_face_landmarks)
            smoothed_yaw   = 0.2 * raw_yaw   + 0.8 * smoothed_yaw
            smoothed_pitch = 0.2 * raw_pitch + 0.8 * smoothed_pitch
            smoothed_roll  = 0.2 * raw_roll  + 0.8 * smoothed_roll

        cam_aspect = rgb.shape[0] / max(1, rgb.shape[1])

        loop_state["menu_buttons"] = render_scene(
            loop_state["screen"], rgb, loop_state, cam_aspect,
            smoothed_x, smoothed_y, smoothed_yaw, smoothed_pitch, smoothed_roll,
            current_face_landmarks, has_face, clock.get_fps())

        if handle_events(pygame.event.get(), loop_state):
            break

        if loop_state["runtime_settings"]["fpsLimit"] > 0:
            clock.tick(loop_state["runtime_settings"]["fpsLimit"])
        else:
            clock.tick()

except KeyboardInterrupt:
    pass
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    pygame.quit()
    sys.exit()