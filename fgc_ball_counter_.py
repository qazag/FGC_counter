import argparse
import base64
import csv
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, fields

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "goals.json")
WINDOW = "FGC ball counter"


@dataclass
class Settings:
    proc_max_side: int = 960
    auto_color: bool = True
    hue_lo: int = 5
    hue_hi: int = 20
    sat_lo: int = 100
    val_lo: int = 70
    ball_radius: float = 0.0
    motion_thr: int = 18
    stabilize: bool = True
    moving_camera: bool = False
    lock_basket: bool = True
    min_rise: float = 8.0
    min_rise_speed: float = 1.2
    max_missed: int = 8
    vanish_frames: int = 3
    confirm_inside: int = 2
    undo_sec: float = 0.8
    learn_shots: int = 3
    cluster_radius: float = 8.0
    n_goals: int = 1
    auto_human: bool = True
    human_color_frac: float = 0.12
    snap_structure: bool = True
    auto_red_box: bool = True
    apriltag: bool = True
    tag_family: str = "36h11"
    tag_ids: str = ""
    web_port: int = 8080


def _quiet_opencv():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


def _backends():
    system = platform.system()
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if system == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


AUTO_EXPOSURE_ON = {cv2.CAP_DSHOW: 0.75, cv2.CAP_V4L2: 3}


def frame_brightness(frame):
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def open_camera(index, width=None, height=None, fps=None):
    best = None
    for backend in _backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if backend in AUTO_EXPOSURE_ON:
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_ON[backend])
            except cv2.error:
                pass
        bright, frame = -1.0, None
        for _ in range(30):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                bright = frame_brightness(f)
                if bright > 25:
                    break
        if frame is None:
            cap.release()
            continue
        if bright > 25:
            if best is not None:
                best[0].release()
            return cap
        if best is None or bright > best[1]:
            if best is not None:
                best[0].release()
            best = (cap, bright)
        else:
            cap.release()
    if best is not None:
        print(DARK_HINT)
        return best[0]
    return None


DARK_HINT = ("  ВНИМАНИЕ: картинка с камеры тёмная/чёрная. Проверьте:\n"
             "   1) не закрыт ли объектив или шторка камеры;\n"
             "   2) не занята ли камера другой программой (Zoom, Teams, браузер, OBS) - закройте их;\n"
             "   3) та ли камера - попробуйте --camera 1 или нажмите N в окне;\n"
             "   4) Параметры Windows -> Конфиденциальность -> Камера: разрешить приложениям.")


def list_cameras(max_index=8):
    _quiet_opencv()
    found = []
    for i in range(max_index):
        cap = open_camera(i)
        if cap is None:
            continue
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            found.append((i, w, h))
        cap.release()
    return found


def pick_camera(max_index=8):
    cams = list_cameras(max_index)
    if not cams:
        return None, []
    best = max(cams, key=lambda c: (c[1] * c[2], c[0]))
    return best[0], cams


class Source:
    def __init__(self, spec, args):
        self.args = args
        self.live = True
        self.cam_index = None
        self.name = str(spec)
        if spec is None:
            idx, cams = pick_camera()
            if idx is None:
                sys.exit("Камера не найдена. Подключите камеру или укажите видеофайл.")
            print("Камеры:", ", ".join(f"#{i} {w}x{h}" for i, w, h in cams), f"-> беру #{idx}")
            spec = idx
        if isinstance(spec, int) or str(spec).isdigit():
            self.cam_index = int(spec)
            self.cap = open_camera(self.cam_index, args.width, args.height, args.fps)
            self.name = f"камера #{self.cam_index}"
            if self.cap is None:
                sys.exit(f"Не открывается {self.name}")
        else:
            path = str(spec)
            self.live = "://" in path
            if not self.live and not os.path.exists(path) and os.path.exists(os.path.join(HERE, path)):
                path = os.path.join(HERE, path)
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                sys.exit(f"Не открывается источник: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1.0 < fps <= 240.0 else 30.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.n_frames = n if not self.live and n > 0 else 0

    def read(self):
        return self.cap.read()

    def next_camera(self):
        if self.cam_index is None:
            return False
        for step in range(1, 9):
            idx = (self.cam_index + step) % 9
            cap = open_camera(idx, self.args.width, self.args.height, self.args.fps)
            if cap is not None:
                self.cap.release()
                self.cap, self.cam_index, self.name = cap, idx, f"камера #{idx}"
                return True
        return False

    def release(self):
        self.cap.release()


class CameraMotion:
    SMALL_W = 240

    def __init__(self, enabled, follow=False):
        self.enabled = enabled
        self.follow = follow
        self.ref = None
        self.prev = None
        self.D = np.zeros(2)
        self.rel = np.zeros(2)
        self.resp = 1.0
        self.keys = []
        self.n = 0

    def _small(self, frame):
        h, w = frame.shape[:2]
        self.k = w / float(self.SMALL_W)
        g = cv2.cvtColor(cv2.resize(frame, (self.SMALL_W, int(round(h / self.k)))), cv2.COLOR_BGR2GRAY)
        return g.astype(np.float32)

    def set_reference(self, frame=None, small_gray=None):
        self.ref = small_gray.astype(np.float32) if small_gray is not None else self._small(frame)
        self.win = cv2.createHanningWindow((self.ref.shape[1], self.ref.shape[0]), cv2.CV_32F)
        self.prev = None
        self.D = np.zeros(2)
        self.rel = np.zeros(2)
        self.keys = [(self.ref, np.zeros(2))]

    def match(self, frame):
        g = self._small(frame)
        if g.shape != self.ref.shape:
            return 0.0, np.zeros(2)
        (dx, dy), resp = cv2.phaseCorrelate(self.ref, g, self.win)
        return float(resp), np.array([dx, dy]) * self.k

    def update(self, frame):
        self.n += 1
        if not self.enabled:
            return self.rel, self.D
        g = self._small(frame)
        if self.ref is None or g.shape != self.ref.shape:
            self.set_reference(small_gray=g)
        if self.prev is None:
            self.prev = g
            return self.rel, self.D
        (dx, dy), resp = cv2.phaseCorrelate(self.prev, g, self.win)
        self.resp = float(resp)
        rel = np.array([dx, dy]) * self.k if resp >= 0.03 else np.zeros(2)
        if float(np.abs(rel).max()) < 0.15 * self.k:
            rel = np.zeros(2)
        self.rel = rel
        self.prev = g
        if not self.follow:
            return self.rel, self.D
        self.D = self.D + rel
        if self.n % 5 == 0:
            self._correct(g)
        return self.rel, self.D

    def _correct(self, g):
        w = g.shape[1] * self.k
        best = min(self.keys, key=lambda kd: float(np.abs(kd[1] - self.D).max()))
        if float(np.abs(best[1] - self.D).max()) < 0.35 * w:
            (dx, dy), resp = cv2.phaseCorrelate(best[0], g, self.win)
            if resp >= 0.25:
                D = best[1] + np.array([dx, dy]) * self.k
                if float(np.abs(D - self.D).max()) < 0.1 * w:
                    self.D = 0.5 * self.D + 0.5 * D
        if float(np.abs(best[1] - self.D).max()) > 0.25 * w and len(self.keys) < 60:
            self.keys.append((g.copy(), self.D.copy()))

    def set_D(self, D, weight=0.5):
        D = np.asarray(D, np.float64)
        step = float(np.abs(D - self.D).max())
        if step < 2.0:
            self._jump = None
            return
        if step > 30.0:
            if getattr(self, "_jump", None) is not None and float(np.abs(D - self._jump[0]).max()) < 6:
                self._jump = (D, self._jump[1] + 1)
            else:
                self._jump = (D, 1)
            if self._jump[1] < 3:
                return
            self._jump, self.D = None, D
            return
        self._jump = None
        self.D = (1 - weight) * self.D + weight * D


class BallModel:
    def __init__(self, s: Settings, frame_h):
        self.lo = np.array([s.hue_lo, s.sat_lo, s.val_lo], np.uint8)
        self.hi = np.array([s.hue_hi, 255, 255], np.uint8)
        self.auto_color = s.auto_color
        self.fitted = not s.auto_color
        self.hues, self.sats = [], []
        self.default_r = max(3.0, frame_h / 140.0)
        self.r = s.ball_radius if s.ball_radius > 0 else self.default_r
        self.auto_r = s.ball_radius <= 0
        self.r_samples = []

    def mask(self, hsv):
        return cv2.inRange(hsv, self.lo, self.hi)

    @property
    def area(self):
        return math.pi * self.r * self.r

    def learn_color(self, hsv, moving):
        if self.fitted:
            return
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        sel = (S > 150) & (V > 100) & (H >= 4) & (H <= 30)
        if np.count_nonzero(sel) > 50:
            self.hues.append(H[sel][::4])
            self.sats.append(S[sel & moving] if np.count_nonzero(sel & moving) else S[sel][:0])
        if sum(len(h) for h in self.hues) < 20000 and len(self.hues) < 60:
            return
        if not self.hues:
            return
        H = np.concatenate(self.hues)
        hist = np.bincount(H, minlength=31)[4:31]
        h0 = int(np.argmax(hist)) + 4
        self.lo = np.array([max(5, h0 - 4), 90, 60], np.uint8)
        self.hi = np.array([min(34, h0 + 7), 255, 255], np.uint8)
        self.fitted = True
        self.hues, self.sats = [], []
        print(f"  [цвет] мяч: оттенок {h0} -> H {self.lo[0]}..{self.hi[0]}, S > {self.lo[1]}")

    def learn_radius(self, blobs):
        if not self.auto_r:
            return
        for area, w, h in blobs:
            ar = min(w, h) / max(1, max(w, h))
            fill = area / max(1, w * h)
            if ar > 0.75 and fill > 0.6:
                self.r_samples.append(math.sqrt(area / math.pi))
        if len(self.r_samples) >= 40:
            r = float(np.median(self.r_samples))
            self.r = float(np.clip(r, 0.5 * self.default_r, 2.5 * self.default_r))
            self.auto_r = False
            print(f"  [размер] радиус мяча = {self.r:.1f} px")

    def to_json(self, frame_h):
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist(),
                "radius_norm": round(self.r / frame_h, 5)}

    def load(self, d, frame_h):
        self.lo = np.array(d["lo"], np.uint8)
        self.hi = np.array(d["hi"], np.uint8)
        self.fitted = True
        if d.get("radius_norm"):
            self.r = float(d["radius_norm"]) * frame_h
            self.auto_r = False


class Track:
    def __init__(self, tid, img, world, fi):
        self.id = tid
        self.img = (float(img[0]), float(img[1]))
        self.pts = [(fi, float(world[0]), float(world[1]))]
        self.own = [(fi, float(world[0]), float(world[1]))]
        self.vx = self.vy = 0.0
        self.missed = 0
        self.flight = False
        self.counted = -1
        self.counted_frame = -1
        self.inside_run = 0
        self.parent = None

    @property
    def x(self):
        return self.pts[-1][1]

    @property
    def y(self):
        return self.pts[-1][2]

    @property
    def age(self):
        return len(self.pts)

    def base(self, bg):
        k = self.missed + 1
        bx, by = bg(self.img)
        return self.img[0] + bx * k, self.img[1] + by * k

    def predict(self, bg):
        k = self.missed + 1
        bx, by = self.base(bg)
        return bx + self.vx * k, by + self.vy * k

    def add(self, img, world, base, fi):
        k = max(1, fi - self.pts[-1][0])
        ox, oy = img[0] - base[0], img[1] - base[1]
        nvx, nvy = ox / k, oy / k
        if self.age == 1:
            self.vx, self.vy = nvx, nvy
        else:
            self.vx = 0.6 * nvx + 0.4 * self.vx
            self.vy = 0.6 * nvy + 0.4 * self.vy
        self.img = (float(img[0]), float(img[1]))
        self.pts.append((fi, float(world[0]), float(world[1])))
        _, px, py = self.own[-1]
        self.own.append((fi, px + ox, py + oy))
        if len(self.pts) > 90:
            self.pts = self.pts[:1] + self.pts[-89:]
            self.own = self.own[:1] + self.own[-89:]
        self.missed = 0

    def check_flight(self, r, s: Settings, min_pts=4):
        if self.flight or self.age < 3:
            return self.flight
        pts = self.own
        start = 0
        for j in range(1, len(pts)):
            if pts[j][2] >= pts[j - 1][2] or pts[j][0] - pts[j - 1][0] > 2:
                start = j
                continue
            if j - start >= min_pts - 1:
                f0, _, y0 = pts[start]
                fj, _, yj = pts[j]
                rise = y0 - yj
                if rise >= s.min_rise * r and rise / max(1, fj - f0) >= s.min_rise_speed * r:
                    self.flight = True
                    return True
        return False

    def descending(self, r):
        min_y = min(p[2] for p in self.own)
        return self.vy > 0.2 * r or self.own[-1][2] > min_y + r


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.forbid = None

    def update(self, dets, D, bg, r, fi, max_missed, link_missed=3, fps_k=1.0):
        pairs = []
        bases = {}
        for ti, t in enumerate(self.tracks):
            if t.missed > link_missed:
                continue
            bx, by = bases[ti] = t.base(bg)
            px, py = t.predict(bg)
            speed = math.hypot(t.vx, t.vy)
            k = t.missed + 1
            if t.age == 1:
                gate_pred, gate_last = 9.0 * r * fps_k, 0.0
            else:
                gate_pred = 3.0 * r + 0.8 * speed + r * t.missed
                gate_last = (1.3 * speed + 3.0 * r) * k if t.vy < 0.5 * r else 0.0
            for di, (x, y) in enumerate(dets):
                if t.vy > r and y < by - 2 * r:
                    continue
                if self.forbid is not None and self.forbid(t, x - D[0], y - D[1]):
                    continue
                if t.vy > 0.5 * r and t.age >= 3 and \
                        math.hypot(x - bx, y - by) > 1.4 * speed * k + 2 * r:
                    continue
                dp = math.hypot(x - px, y - py)
                dl = math.hypot(x - bx, y - by)
                if dp <= gate_pred:
                    pairs.append((dp, ti, di))
                elif dl <= gate_last:
                    pairs.append((1.5 * dl + r, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            x, y = dets[di]
            self.tracks[ti].add((x, y), (x - D[0], y - D[1]), bases[ti], fi)
        dead, alive = [], []
        for ti, t in enumerate(self.tracks):
            t.seen = ti in used_t
            if not t.seen:
                t.missed += 1
                if t.missed > max_missed:
                    dead.append(t)
                    continue
            alive.append(t)
        lost = [t for t in alive if t.flight and not t.seen and t.missed <= link_missed
                and t.counted < 0]
        for di, (x, y) in enumerate(dets):
            if di not in used_d:
                t = Track(self.next_id, (x, y), (x - D[0], y - D[1]), fi)
                t.seen = True
                self.next_id += 1
                best = None
                for p in lost:
                    reach = max(2.5 * math.hypot(p.vx, p.vy), 12.0 * r) * (p.missed + 1) * fps_k
                    d = math.hypot(x - p.img[0], y - p.img[1])
                    if d > reach or y - D[1] > p.pts[0][2] - 8 * r:
                        continue
                    if p.vy > r and y < p.img[1] - 2 * r:
                        continue
                    if best is None or d < best[0]:
                        best = (d, p)
                if best is not None:
                    t.flight, t.parent = True, best[1]
                alive.append(t)
        self.tracks = alive
        return dead


class Goal:
    def __init__(self, x0, y0, x1, y1, source="learned", snapped=False):
        self.x0, self.y0, self.x1, self.y1 = float(x0), float(y0), float(x1), float(y1)
        self.source = source
        self.snapped = snapped
        self.score = 0
        self.human = source.startswith("human")
        self.anchor = None

    def armed(self, pts, r):
        return any(y < self.y0 and self.x0 - 5 * r <= x <= self.x1 + 5 * r for _, x, y in pts)

    def inside(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def end_zone(self, x, y, r, side=4.0):
        return self.x0 - side * r <= x <= self.x1 + side * r and self.y0 - 2 * r <= y <= self.y1

    def entered(self, pts, r):
        above = False
        for _, x, y in pts:
            if y < self.y0:
                above = self.x0 - 5 * r <= x <= self.x1 + 5 * r or above
            elif above and self.x0 - r <= x <= self.x1 + r:
                return True
        return False

    def trim(self, pts, r):
        above = entered = False
        for i, (_, x, y) in enumerate(pts):
            if y < self.y0:
                above = above or self.x0 - 5 * r <= x <= self.x1 + 5 * r
            elif above and self.x0 - r <= x <= self.x1 + r:
                entered = True
            if entered and self.side_jump(x, y, r):
                return pts[:i]
        return pts

    def side_jump(self, x, y, r):
        return y <= self.y1 and not (self.x0 - 4 * r <= x <= self.x1 + 4 * r)

    def to_json(self, w, h):
        return {"bbox_norm": [round(self.x0 / w, 5), round(self.y0 / h, 5),
                              round(self.x1 / w, 5), round(self.y1 / h, 5)],
                "source": self.source, "snapped": self.snapped}

    @classmethod
    def from_json(cls, d, w, h):
        x0, y0, x1, y1 = d["bbox_norm"]
        return cls(x0 * w, y0 * h, x1 * w, y1 * h, d.get("source", "saved"), d.get("snapped", False))


def snap_to_structure(frame, model, pts, top, r):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ball = (H >= model.lo[0]) & (H <= model.hi[0])
    colored = ((S > 80) & (V > 40) & ~ball).astype(np.uint8) * 255
    colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(colored, 8)
    h, w = colored.shape
    cx = float(np.median(pts[:, 0]))
    best = None
    for k in range(2, 12):
        for dx in (0, -r, r):
            x, y = int(cx + dx), int(top + k * r)
            if 0 <= x < w and 0 <= y < h and lab[y, x] > 0:
                best = lab[y, x]
                break
        if best:
            break
    if not best:
        return None
    x, y, bw, bh, area = stats[best]
    if area < 40 * math.pi * r * r or bh < 5 * r or abs(y - top) > 8 * r:
        return None
    inside = np.mean((pts[:, 0] >= x - 4 * r) & (pts[:, 0] <= x + bw + 4 * r))
    if inside < 0.6:
        return None
    return x, y, x + bw, y + bh


def build_goal(end_pts, frame, model, s: Settings, D=(0.0, 0.0)):
    r = model.r
    pts = np.array(end_pts, np.float32)
    top = float(pts[:, 1].min() - r)
    if s.snap_structure and frame is not None:
        box = snap_to_structure(frame, model, pts + np.float32(D), top + D[1], r)
        if box:
            x0, y0, x1, y1 = box
            return Goal(x0 - D[0], y0 - D[1], x1 - D[0], y1 - D[1], snapped=True)
    x0 = pts[:, 0].min() - 2 * r
    x1 = pts[:, 0].max() + 2 * r
    h = frame.shape[0] if frame is not None else top + 20 * r
    y1 = min(h - 1, max(pts[:, 1].max() + 2 * r, top + 6 * r))
    return Goal(x0, top, x1, y1)


def overlap(a, b):
    iw = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    ih = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    small = min((a.x1 - a.x0) * (a.y1 - a.y0), (b.x1 - b.x0) * (b.y1 - b.y0))
    return iw * ih / max(1.0, small)


def clusters(points, radius):
    pts = np.array(points, np.float32)
    left = list(range(len(pts)))
    out = []
    while left:
        sub = pts[left]
        d = np.linalg.norm(sub[:, None] - sub[None], axis=2)
        cnt = (d <= radius).sum(1)
        c = int(np.argmax(cnt))
        members = [left[i] for i in np.nonzero(d[c] <= radius)[0]]
        out.append(members)
        left = [i for i in left if i not in members]
    return out


def _look(frame):
    return cv2.GaussianBlur(frame, (3, 3), 0)


def match_tpl(g, tpl, pos, rad, q):
    H, W = g.shape[:2]
    th, tw = tpl.shape[:2]
    cx0, cy0 = int(max(0, -(pos[0] - rad))), int(max(0, -(pos[1] - rad)))
    cx1, cy1 = int(min(tw, W - (pos[0] + rad))), int(min(th, H - (pos[1] + rad)))
    if cx1 - cx0 < 0.4 * tw or cy1 - cy0 < 0.4 * th:
        cx0, cy0 = int(max(0, -pos[0])), int(max(0, -pos[1]))
        cx1, cy1 = int(min(tw, W - pos[0])), int(min(th, H - pos[1]))
    if cx1 - cx0 < 0.4 * tw or cy1 - cy0 < 0.4 * th:
        return None
    t = tpl[cy0:cy1, cx0:cx1]
    px, py = pos[0] + cx0, pos[1] + cy0
    sx0, sy0 = int(max(0, px - rad)), int(max(0, py - rad))
    sx1, sy1 = int(min(W, px + t.shape[1] + rad)), int(min(H, py + t.shape[0] + rad))
    win = g[sy0:sy1, sx0:sx1]
    if q < 1.0:
        t = cv2.resize(t, None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
        win = cv2.resize(win, None, fx=q, fy=q, interpolation=cv2.INTER_AREA)
    if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1] or min(t.shape[:2]) < 6 or float(t.std()) < 3:
        return None
    res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
    _, sc, _, (mx, my) = cv2.minMaxLoc(res)
    fx, fy = float(mx), float(my)
    if 0 < mx < res.shape[1] - 1:
        a, b, c = res[my, mx - 1], res[my, mx], res[my, mx + 1]
        if a - 2 * b + c < 0:
            fx += 0.5 * (a - c) / (a - 2 * b + c)
    if 0 < my < res.shape[0] - 1:
        a, b, c = res[my - 1, mx], res[my, mx], res[my + 1, mx]
        if a - 2 * b + c < 0:
            fy += 0.5 * (a - c) / (a - 2 * b + c)
    return np.array([sx0 + fx / q - cx0, sy0 + fy / q - cy0]), float(sc)


class Anchor:
    def __init__(self, tpl, off, L, s0):
        self.tpl = tpl
        self.off = np.asarray(off, np.float64)
        self.L = np.asarray(L, np.float64)
        self.s0 = float(s0)
        self.pos = None
        self.k = 1.0
        self.score = 0.0
        self.seen = False
        self.pend = None
        self.miss = 0
        self._cache = (1.0, tpl)

    def tpl_at(self, k):
        if abs(k - 1.0) < 1e-3:
            return self.tpl
        if abs(self._cache[0] - k) > 1e-4:
            self._cache = (k, cv2.resize(self.tpl, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR))
        return self._cache[1]

    def box(self):
        o = self.off * self.k
        return (self.pos[0] + o[0], self.pos[1] + o[1], self.pos[0] + o[2], self.pos[1] + o[3])


class BasketLock:
    COARSE, FINE = 64, 160
    GOOD, OK = 0.6, 0.45

    def __init__(self):
        self.items = []
        self.S = 1.0
        self.t = np.zeros(2)
        self.lost = 0
        self.n = 0
        self.best = 0.0
        self._jump = None

    def add(self, frame, box, r=6.0):
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = box
        m = max(0.3 * max(x1 - x0, y1 - y0), 8 * r)
        tx0, ty0 = int(max(0, x0 - m)), int(max(0, y0 - m))
        tx1, ty1 = int(min(w, x1 + m)), int(min(h, y1 + m))
        if tx1 - tx0 < 12 or ty1 - ty0 < 12:
            return None
        tpl = _look(frame)[ty0:ty1, tx0:tx1].copy()
        a = Anchor(tpl, [x0 - tx0, y0 - ty0, x1 - tx0, y1 - ty0],
                   (np.array([tx0, ty0], np.float64) - self.t) / self.S, self.S)
        a.pos, a.seen, a.score = np.array([tx0, ty0], np.float64), True, 1.0
        self.items.append(a)
        return a

    def remove(self, a):
        if a in self.items:
            self.items.remove(a)

    def _pred(self, a, t):
        return self.S * a.L + t

    def update(self, frame, rel=(0.0, 0.0)):
        if not self.items:
            return
        self.n += 1
        g = _look(frame)
        H, W = g.shape[:2]
        rel = np.asarray(rel, np.float64)
        t_old = self.t.copy()
        t_pred = self.t + rel
        rad = (0.3 if self.lost > 3 else 0.1) * max(W, H) + float(np.abs(rel).max())
        tol = 0.04 * max(W, H)
        hits = []
        for a in self.items:
            a.k = self.S / a.s0
            tpl = a.tpl_at(a.k)
            p = self._pred(a, t_pred)
            qc = min(1.0, self.COARSE / float(max(tpl.shape[:2])))
            h = match_tpl(g, tpl, p, rad, qc)
            if h is not None and h[1] >= self.OK:
                qf = min(1.0, self.FINE / float(max(tpl.shape[:2])))
                f = match_tpl(g, tpl, h[0], 2.0 / qc + 2, qf)
                if f is not None and f[1] >= h[1] - 0.05:
                    h = f
            if h is not None and self._visible(a, h[0], W, H) < 0.5:
                h = (h[0], min(h[1], self.GOOD - 0.01))
            a.score = h[1] if h is not None else 0.0
            hits.append(h)
        order = sorted(range(len(self.items)), key=lambda i: -self.items[i].score)
        best = order[0]
        t_new = None
        moved = False
        hb = hits[best]
        if hb is not None and hb[1] >= self.OK:
            tb = hb[0] - self.S * self.items[best].L
            near = float(np.abs(tb - t_pred).max()) < tol
            if near or hb[1] >= self.GOOD:
                ts = [tb]
                for i in order[1:]:
                    if hits[i] is not None and hits[i][1] >= self.OK:
                        ti = hits[i][0] - self.S * self.items[i].L
                        if float(np.abs(ti - tb).max()) < tol:
                            ts.append(ti)
                t_new = np.mean(ts, 0)
                if not near and not (len(ts) >= 2 and self.lost > 3):
                    if self._jump is not None and float(np.abs(self._jump - t_new).max()) < tol:
                        self._jump = None
                    else:
                        self._jump = t_new
                        t_new = None
                else:
                    self._jump = None
        if t_new is None:
            self.lost += 1
            moved = float(np.abs(rel).max()) >= 2.0
            if moved:
                self.t = t_pred
            if self.lost > 10 and self.lost % 10 == 0:
                self.find(g, (1.0,))
        else:
            self.lost = 0
            self.t = t_new
        shifted = not self.lost and float(np.abs(self.t - t_old).max()) >= 1.5
        for a, h in zip(self.items, hits):
            p = self._pred(a, self.t)
            ok = h is not None and h[1] >= self.OK and float(np.abs(h[0] - p).max()) < tol
            if ok:
                p = h[0]
                a.L = 0.9 * a.L + 0.1 * (h[0] - self.t) / self.S
            a.seen = ok
            held = t_new is not None and self._visible(a, p, W, H) >= 0.5
            a.miss = 0 if ok or held else a.miss + 1
            step = float(np.abs(p - a.pos).max()) if a.pos is not None else 1e9
            if self.lost and a.pos is not None:
                if moved:
                    a.pos, a.pend = p, None
            elif step >= 3.0 or (shifted and step >= 0.5):
                a.pos, a.pend = p, None
            elif step >= 1.5:
                if a.pend is not None and float(np.abs(p - a.pend).max()) < 1.0:
                    a.pos, a.pend = p, None
                else:
                    a.pend = p
            else:
                a.pend = None

    def _visible(self, a, pos, W, H):
        o = a.off * a.k
        x0, y0, x1, y1 = pos[0] + o[0], pos[1] + o[1], pos[0] + o[2], pos[1] + o[3]
        iw = max(0.0, min(W, x1) - max(0.0, x0))
        ih = max(0.0, min(H, y1) - max(0.0, y0))
        return iw * ih / max(1.0, (x1 - x0) * (y1 - y0))

    def find(self, g, scales=(0.9, 1.0, 1.1), need=0.65):
        H, W = g.shape[:2]
        best = None
        for a in self.items:
            for f in scales:
                k = self.S * f / a.s0
                if not 0.2 < k < 4.0:
                    continue
                tpl = a.tpl_at(k)
                if tpl.shape[0] >= H or tpl.shape[1] >= W:
                    continue
                q = min(1.0, self.COARSE / float(max(tpl.shape[:2])))
                h = match_tpl(g, tpl, (0.0, 0.0), max(W, H), q)
                if h is not None and self._visible(a, h[0], W, H) >= 0.5 and \
                        (best is None or h[1] > best[0]):
                    best = (h[1], a, f, h[0])
        self.best = best[0] if best else 0.0
        if not best or best[0] < need:
            return False
        sc, a, f, p = best
        self.S *= f
        a.k = self.S / a.s0
        fine = match_tpl(g, a.tpl_at(a.k), p, 6, min(1.0, self.FINE / float(max(a.tpl_at(a.k).shape[:2]))))
        if fine is not None and fine[1] >= sc - 0.05:
            p = fine[0]
        self.t = p - self.S * a.L
        self.lost = 0
        return True


TAG_FAMILIES = {"36h11": "DICT_APRILTAG_36h11", "25h9": "DICT_APRILTAG_25h9",
                "16h5": "DICT_APRILTAG_16h5", "36h10": "DICT_APRILTAG_36h10"}


class TagFinder:
    def __init__(self, family="36h11", ids=""):
        self.ok = False
        name = TAG_FAMILIES.get(str(family).replace("tag", ""), str(family))
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, name):
            print("  [AprilTag] в этой версии OpenCV нет AprilTag (нужен opencv-python >= 4.7)")
            return
        dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters()).detectMarkers
        else:
            prm = cv2.aruco.DetectorParameters_create()
            self._det = lambda g: cv2.aruco.detectMarkers(g, dic, parameters=prm)
        self.ids = {int(v) for v in str(ids).replace(" ", "").split(",") if v} or None
        self.ok = True

    def find(self, frame, want_id=None):
        if not self.ok:
            return None
        corners, ids, _ = self._det(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None:
            return None
        best = None
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if (self.ids and i not in self.ids) or (want_id is not None and i != want_id):
                continue
            c = c.reshape(4, 2).astype(np.float32)
            size = math.sqrt(max(1.0, cv2.contourArea(c)))
            if best is None or size > best[2]:
                best = (i, c.mean(0), size, c)
        return best


class BallCounter:
    def __init__(self, s: Settings, fps, frame_shape):
        self.s = s
        self.fps = fps
        self.h, self.w = frame_shape[:2]
        self.model = BallModel(s, self.h)
        self.stab = CameraMotion(s.stabilize, follow=s.moving_camera or s.lock_basket)
        self.human = []
        self.tag = TagFinder(s.tag_family, s.tag_ids) if s.apriltag else None
        self.tag_ref = None
        self.tag_pending = False
        self.tag_seen = None
        self.tracker = Tracker()
        self.lock = BasketLock() if s.lock_basket else None
        self.goals = []
        self.landings = []
        self.events = []
        self.prev_gray = None
        self._still = 0
        self.fi = 0
        self.t = 0.0
        self.flash = -1
        self.last_frame = None
        self.ref_frame = None
        self.ref_D = np.zeros(2)
        self.D = np.zeros(2)
        k = fps / 30.0
        self.fps_k = max(1.0, 1.0 / k)
        self.max_missed = max(3, round(s.max_missed * k))
        self.link_missed = max(2, round(3 * k))
        self.vanish_frames = max(2, round(s.vanish_frames * k))
        self.confirm_inside = max(1, round(s.confirm_inside * k))
        self.min_pts = 4 if fps >= 24 else 3
        self.tracker.forbid = self._forbid

    def _forbid(self, t, x, y):
        if not t.flight or not self.goals:
            return False
        r = self.model.r
        for g in self.goals:
            if g.end_zone(t.x, t.y, r) and t.y >= g.y0 and g.entered(t.pts, r):
                return g.side_jump(x, y, r)
        return False

    def _update_tag(self, frame):
        if self.tag is None or not self.tag.ok:
            return
        recent = self.tag_seen is not None and self.fi - self.tag_seen[0] < 2 * self.fps
        if self.fi % (2 if recent or self.fi < 5 * self.fps else 10):
            return
        want = self.tag_ref["id"] if self.tag_ref else None
        hit = self.tag.find(frame, want)
        if hit is None:
            return
        tid, c, size, corners = hit
        self.tag_seen = (self.fi, corners)
        if self.tag_ref is None:
            self.tag_ref = {"id": tid, "center": (c - self.stab.D).tolist(), "size": size}
            print(f"  [AprilTag] вижу метку id={tid} - теперь рамка держится за неё")
            return
        if self.tag_pending:
            k = size / float(self.tag_ref["size"])
            c0 = np.array(self.tag_ref["center"], np.float64)
            for g in self.goals + self.human:
                g.x0, g.y0 = c + (np.array([g.x0, g.y0]) - c0) * k
                g.x1, g.y1 = c + (np.array([g.x1, g.y1]) - c0) * k
            self.model.r *= k
            for g in self.goals + self.human:
                g.anchor = None
            self.stab.set_reference(frame)
            self.prev_gray = None
            self.tracker = Tracker()
            self.tracker.forbid = self._forbid
            self.tag_ref = {"id": tid, "center": c.tolist(), "size": size}
            self.tag_pending = False
            print(f"  [AprilTag] рамка поставлена по метке id={tid} (масштаб {k:.2f})")
            return
        self.stab.set_D(c - np.array(self.tag_ref["center"]), 0.5)

    def in_human(self, x, y):
        return any(z.inside(x, y) for z in self.human)

    def _auto_red_box(self, frame, D):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        red = (((H >= 168) | (H <= 2)) & (S > 100) & (V > 50)).astype(np.uint8) * 255
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, st, _ = cv2.connectedComponentsWithStats(red, 8)
        best = None
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if area < 0.015 * h * w or bh < 0.1 * h or bw < 0.05 * w or area < 0.5 * bw * bh:
                continue
            if best is None or area > best[4]:
                best = st[i]
        if best is None:
            return
        x, y, bw, bh, _ = best
        pad = 0.08 * bw
        x0, x1 = max(0.0, x - pad), min(float(w - 1), x + bw + pad)
        self.goals = [Goal(x0 - D[0], y - D[1], x1 - D[0], y + bh - D[1], source="auto_red")]
        x, bw = int(x0), int(x1 - x0)
        print(f"  [корзина] красный короб (корзина A) найден сам: x {x}..{x + bw}, y {y}..{y + bh}"
              "  (C - поставить рамку иначе)")

    @staticmethod
    def _twin_counted(tr):
        return getattr(tr, "blocked", False) or (tr.parent is not None and tr.parent.counted >= 0)

    @staticmethod
    def _claim(tr):
        if tr.parent is not None:
            tr.parent.blocked = True

    def _lock_update(self, frame, rel, D):
        if self.lock is None:
            return
        zones = self.goals + self.human
        keep = {id(g.anchor) for g in zones if g.anchor is not None}
        self.lock.items = [a for a in self.lock.items if id(a) in keep]
        for g in zones:
            if g.anchor is None or g.anchor not in self.lock.items:
                g.anchor = self.lock.add(frame, (g.x0 + D[0], g.y0 + D[1], g.x1 + D[0], g.y1 + D[1]),
                                         self.model.r)
        self.lock.update(frame, rel)
        for g in zones:
            if g.anchor is not None:
                x0, y0, x1, y1 = g.anchor.box()
                g.x0, g.y0, g.x1, g.y1 = x0 - D[0], y0 - D[1], x1 - D[0], y1 - D[1]

    def visible(self, g):
        return g.anchor is None or g.anchor.miss <= max(3, int(0.5 * self.fps))

    @property
    def basket_lost(self):
        return self.lock is not None and bool(self.lock.items) and self.lock.lost > 0

    @property
    def total(self):
        return sum(g.score for g in self.goals)

    def _event(self, gi, delta, reason):
        self.goals[gi].score = max(0, self.goals[gi].score + delta)
        self.events.append({"time_s": round(self.t, 2), "frame": self.fi, "goal": gi,
                            "delta": delta, "goal_score": self.goals[gi].score,
                            "total": self.total, "reason": reason})
        if delta > 0:
            self.flash = self.fi + int(0.3 * self.fps)
        m, sec = divmod(self.t, 60)
        print(f"  [{int(m)}:{sec:05.2f}] {'+' if delta > 0 else ''}{delta} корзина {gi}  "
              f"-> {self.total}   ({reason})", flush=True)

    def classify(self, t_pts, descending, flight, v=None):
        if not (flight and descending):
            return -1
        r = self.model.r
        _, x, y = t_pts[-1]
        if self.in_human(x, y):
            return -1
        if v is not None and y < min(g.y0 for g in self.goals):
            t_pts = list(t_pts) + [(t_pts[-1][0] + 1, x + v[0], y + v[1])]
        for gi, g in enumerate(self.goals):
            if not self.visible(g):
                continue
            pts = g.trim(t_pts, r)
            _, ex, ey = pts[-1]
            if g.end_zone(ex, ey, r) and (g.armed(pts, r) or g.entered(pts, r)):
                return gi
            if v is not None and g.x0 - r <= ex <= g.x1 + r and g.y0 - 6 * r <= ey < g.y0 \
                    and v[1] > 0 and g.armed(pts, r):
                return gi
        return -1

    def reset(self):
        for g in self.goals:
            g.score = 0
        self.events = []
        self.landings = []
        for t in self.tracker.tracks:
            t.counted = -1

    def set_goals(self, goals):
        self.goals = goals
        if any(g.source in ("manual", "auto_red") for g in goals):
            self.human = [z for z in self.human if z.source != "human_auto"]

    def relearn(self):
        self.goals = []
        self.landings = []
        self.tracker = Tracker()
        self.tracker.forbid = self._forbid
        self.prev_gray = None
        if self.last_frame is not None:
            self.stab.set_reference(self.last_frame)
        print("  [корзина] ищу заново: жду бросков...")

    def _transparent(self, g):
        if self.ref_frame is None:
            return False
        r = self.model.r
        h, w = self.ref_frame.shape[:2]
        D = self.ref_D
        x0, x1 = int(max(0, g.x0 + D[0] - 2 * r)), int(min(w, g.x1 + D[0] + 2 * r))
        y0, y1 = int(max(0, g.y0 + D[1] - 2 * r)), int(min(h, g.y1 + D[1] + 2 * r))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return False
        hsv = cv2.cvtColor(self.ref_frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        ball = (H >= self.model.lo[0]) & (H <= self.model.hi[0])
        colored = (S > 90) & (V > 40) & ~ball
        frac = float(np.count_nonzero(colored)) / max(1, np.count_nonzero(~ball))
        return frac < self.s.human_color_frac

    def _try_learn(self):
        s = self.s
        r = self.model.r
        if not self.landings or any(g.source in ("manual", "auto_red") for g in self.goals):
            return
        ends = [(p[-1][1], p[-1][2]) for _, p, _, _ in self.landings]
        usable = [i for i, (_, p, _, _) in enumerate(self.landings)
                  if p[-1][2] < p[0][2] - 8 * r and p[0][2] - min(q[2] for q in p) >= 12 * r
                  and not self.in_human(p[-1][1], p[-1][2])]
        groups = [[usable[i] for i in g]
                  for g in clusters([ends[i] for i in usable], s.cluster_radius * r)
                  if len(g) >= s.learn_shots] if usable else []
        if not groups:
            return
        manual = any(g.source == "manual" for g in self.goals)
        changed, rebuilt = False, False
        for grp in groups[:4]:
            pts = [ends[i] for i in grp]
            cx, cy = float(np.median([p[0] for p in pts])), float(np.median([p[1] for p in pts]))
            if any(g.end_zone(cx, cy, r) for g in self.goals):
                continue
            if self.in_human(cx, cy):
                continue
            g = build_goal(pts, self.ref_frame, self.model, s, self.ref_D)
            if s.auto_human and self._transparent(g):
                g.source, g.human = "human_auto", True
                self.human.append(g)
                print(f"  [корзина] прозрачная корзина (для людей) - НЕ считаю: "
                      f"x {g.x0:.0f}..{g.x1:.0f}, y {g.y0:.0f}..{g.y1:.0f}")
                changed = True
            elif not manual and len(self.goals) < s.n_goals:
                self.goals.append(g)
                print(f"  [корзина] корзина робота найдена: x {g.x0:.0f}..{g.x1:.0f}, "
                      f"y {g.y0:.0f}..{g.y1:.0f}{' (по коробу)' if g.snapped else ''}")
                changed = True
        if not changed or not self.goals:
            return
        self.events = [e for e in self.events if e["reason"] == "вручную"]
        for g in self.goals:
            g.score = 0
        for k, (t, pts, _, v) in enumerate(self.landings):
            gi = self.classify(pts, True, True, v)
            self.landings[k] = (t, pts, gi, v)
            if gi >= 0:
                t_now = self.t
                self.t = t
                self._event(gi, +1, "пересчёт по новой рамке" if rebuilt else "бросок до появления рамки")
                self.t = t_now
        for tr in self.tracker.tracks:
            tr.counted = -1

    FLOW_SIDE = 240

    def _align_prev(self, gray, rel):
        h, w = gray.shape
        prev, valid = self.prev_gray, None
        if np.abs(rel).max() > 0:
            M = np.float32([[1, 0, rel[0]], [0, 1, rel[1]]])
            prev = cv2.warpAffine(self.prev_gray, M, (w, h))
            valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h), borderValue=0)
        self._still = self._still + 1 if not np.abs(rel).max() > 0 else 0
        if self._still >= 5:
            self._prev_small = None
        elif self.s.stabilize:
            q = min(1.0, self.FLOW_SIDE / float(max(h, w)))
            size = (int(w * q), int(h * q))
            cs = self._no_balls(cv2.resize(gray, size, interpolation=cv2.INTER_AREA),
                                cv2.resize(self.ball_mask, size, interpolation=cv2.INTER_NEAREST))
            ps = self._prev_small if getattr(self, "_prev_small", None) is not None and \
                self._prev_small.shape == cs.shape else cs
            self._prev_small = cs
            fb = cv2.calcOpticalFlowFarneback(cs, ps, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            self.bg = lambda p, fb=fb, q=q: self._bg_at(fb, q, p)
        return prev, valid

    @staticmethod
    def _no_balls(g, mask):
        if not np.any(mask):
            return g
        m = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
        blurred = cv2.blur(g, (15, 15))
        out = g.copy()
        out[m] = blurred[m]
        return out

    def _bg_at(self, fb, q, p):
        hh, ww = fb.shape[:2]
        rad = max(6.0 * self.model.r, 20.0) * q
        xs, ys = [], []
        for a in range(8):
            ang = a * math.pi / 4
            x = int(round(p[0] * q + rad * math.cos(ang)))
            y = int(round(p[1] * q + rad * math.sin(ang)))
            if 0 <= x < ww and 0 <= y < hh:
                fx, fy = fb[y, x]
                xs.append(fx)
                ys.append(fy)
        if not xs:
            return 0.0, 0.0
        return -float(np.median(xs)) / q, -float(np.median(ys)) / q

    def process(self, frame, t):
        s = self.s
        self.fi += 1
        self.t = t
        self.last_frame = frame
        rel, D = self.stab.update(frame)
        self._update_tag(frame)
        D = self.stab.D.copy()
        self.D = D.copy()
        if not self.goals and self.s.auto_red_box and self.fi % 15 == 1:
            self._auto_red_box(frame, D)
        self._lock_update(frame, rel, D)
        if self.fi % 15 == 1:
            self.ref_frame, self.ref_D = frame, D.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = []
        self.bg = lambda p: (0.0, 0.0)
        self.ball_mask = self.model.mask(hsv)
        if self.prev_gray is None and self.s.stabilize:
            self._prev_small = None
        if self.prev_gray is not None:
            prev, valid = self._align_prev(gray, rel)
            moving = cv2.absdiff(gray, prev) > s.motion_thr
            if valid is not None:
                moving &= cv2.erode(valid, np.ones((9, 9), np.uint8)) > 0
            self.model.learn_color(hsv, moving)
            r = self.model.r
            k = max(3, int(2 * r) | 1)
            mv = cv2.dilate(moving.astype(np.uint8) * 255,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            m = cv2.bitwise_and(self.ball_mask, mv)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, _, st, cn = cv2.connectedComponentsWithStats(m, 8)
            a = self.model.area
            blobs = []
            for i in range(1, n):
                area = st[i, cv2.CC_STAT_AREA]
                if 0.25 * a <= area <= 10 * a:
                    dets.append((float(cn[i][0]), float(cn[i][1])))
                    blobs.append((area, st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]))
            self.model.learn_radius(blobs)
        self.prev_gray = gray

        r = self.model.r
        dead = self.tracker.update(dets, D, self.bg, r, self.fi, self.max_missed,
                                   self.link_missed, self.fps_k)
        undo_frames = int(s.undo_sec * self.fps)

        for tr in self.tracker.tracks:
            tr.check_flight(r, s, self.min_pts)
            if not self.goals or not tr.flight or self._twin_counted(tr):
                continue
            if tr.seen:
                gi = next((i for i, g in enumerate(self.goals)
                           if g.inside(tr.x, tr.y) and self.visible(g)), -1)
                if gi >= 0 and self.in_human(tr.x, tr.y):
                    gi = -1
                tr.inside_run = tr.inside_run + 1 if gi >= 0 else 0
                if tr.counted < 0 and gi >= 0 and tr.inside_run >= self.confirm_inside \
                        and tr.descending(r) and self.goals[gi].armed(tr.pts, r):
                    tr.counted, tr.counted_frame = gi, self.fi
                    self._claim(tr)
                    self._event(gi, +1, f"мяч #{tr.id} упал в корзину")
                elif tr.counted >= 0 and self.fi - tr.counted_frame <= undo_frames:
                    g = self.goals[tr.counted]
                    if not g.end_zone(tr.x, tr.y, r, side=6.0) or tr.y > g.y1 + 3 * r:
                        self._event(tr.counted, -1, f"мяч #{tr.id} вылетел")
                        tr.counted = -1
            elif tr.missed == self.vanish_frames and tr.counted < 0:
                gi = self.classify(tr.pts, tr.descending(r), tr.flight, (tr.vx, tr.vy))
                if gi >= 0:
                    tr.counted, tr.counted_frame = gi, self.fi
                    self._claim(tr)
                    self._event(gi, +1, f"мяч #{tr.id} пропал в корзине")

        for tr in dead:
            if not tr.flight or not tr.descending(r) or self._twin_counted(tr):
                continue
            gi = self.classify(tr.pts, True, True, (tr.vx, tr.vy)) if self.goals else -1
            if self.goals:
                if tr.counted >= 0 and gi < 0 and \
                        not self.goals[tr.counted].end_zone(tr.x, tr.y, r, side=6.0):
                    self._event(tr.counted, -1, f"мяч #{tr.id} не в корзине")
                elif tr.counted < 0 and gi >= 0:
                    self._claim(tr)
                    self._event(gi, +1, f"мяч #{tr.id} в корзине")
            self.landings.append((self.t, tr.pts, gi, (tr.vx, tr.vy)))
            self._try_learn()
        return D

    def _load_look(self, data, first_frame):
        goals = [Goal.from_json(g, self.w, self.h) for g in data["goals"]]
        zones = {"goal": goals, "human": self.human}
        lock = BasketLock()
        lock.S = float(data["look"].get("S", 1.0))
        for it in data["look"]["items"]:
            tpl = cv2.imdecode(np.frombuffer(base64.b64decode(it["tpl"]), np.uint8), cv2.IMREAD_COLOR)
            z = zones.get(it.get("kind"), [])
            if tpl is None or not 0 <= int(it.get("i", -1)) < len(z):
                continue
            a = Anchor(tpl, it["off"], it["L"], it["s0"])
            z[int(it["i"])].anchor = a
            lock.items.append(a)
        if not lock.items:
            return False
        saved_w = float((data.get("size") or [self.w])[0])
        k = self.w / saved_w if saved_w > 0 else 1.0
        found = lock.find(_look(first_frame), [k * f for f in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6)])
        self.goals, self.lock = goals, lock
        for g in goals + self.human:
            if g.anchor is not None:
                g.anchor.k = lock.S / g.anchor.s0
                g.anchor.pos = lock.S * g.anchor.L + lock.t
                g.x0, g.y0, g.x1, g.y1 = g.anchor.box()
        if found:
            print(f"  [корзина] корзины найдены по их виду (похожесть {lock.best:.2f}) - рамки на месте")
        else:
            lock.lost = 11
            print("  [корзина] сохранённых корзин пока не видно - ищу их в кадре "
                  "(наведите камеру на корзину; C - задать заново)")
        return True

    def to_json(self):
        ref = self.stab.ref
        data = {"size": [self.w, self.h], "ball": self.model.to_json(self.h),
                "goals": [g.to_json(self.w, self.h) for g in self.goals],
                "human_zones": [g.to_json(self.w, self.h) for g in self.human]}
        if self.tag_ref is not None:
            data["tag"] = {"id": int(self.tag_ref["id"]),
                           "center_norm": [float(self.tag_ref["center"][0]) / self.w,
                                           float(self.tag_ref["center"][1]) / self.h],
                           "size_norm": float(self.tag_ref["size"]) / self.h}
        if ref is not None:
            ok, buf = cv2.imencode(".jpg", ref.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])
            data["reference"] = base64.b64encode(buf.tobytes()).decode("ascii")
        if self.lock is not None and self.lock.items:
            items = []
            for kind, zones in (("goal", self.goals), ("human", self.human)):
                for i, g in enumerate(zones):
                    a = g.anchor
                    if a is None or a not in self.lock.items:
                        continue
                    ok, buf = cv2.imencode(".jpg", a.tpl, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    items.append({"kind": kind, "i": i, "off": [round(float(v), 2) for v in a.off],
                                  "L": [round(float(v), 2) for v in a.L], "s0": round(a.s0, 5),
                                  "tpl": base64.b64encode(buf.tobytes()).decode("ascii")})
            if items:
                data["look"] = {"S": round(self.lock.S, 5), "items": items}
        return data

    def load(self, data, first_frame):
        if data.get("reference"):
            ref = cv2.imdecode(np.frombuffer(base64.b64decode(data["reference"]), np.uint8),
                               cv2.IMREAD_GRAYSCALE)
            if ref is None or float(ref.mean()) < 25:
                print("  goals.json сохранён, когда камера была ЧЁРНОЙ - его цвет мяча и корзину "
                      "не использую, подбираю заново")
                return False
        if "ball" in data:
            self.model.load(data["ball"], self.h)
        self.human = [Goal.from_json(g, self.w, self.h) for g in data.get("human_zones", [])]
        for z in self.human:
            z.human = True
        if data.get("look") and data.get("goals") and self.lock is not None:
            if self._load_look(data, first_frame):
                return True
        if data.get("tag") and data.get("goals") and self.tag is not None and self.tag.ok:
            t = data["tag"]
            self.tag_ref = {"id": int(t["id"]), "size": float(t["size_norm"]) * self.h,
                            "center": [t["center_norm"][0] * self.w, t["center_norm"][1] * self.h]}
            self.tag_pending = True
            self.goals = [Goal.from_json(g, self.w, self.h) for g in data["goals"]]
            print(f"  [AprilTag] корзина сохранена относительно метки id={t['id']} - ищу метку")
            return True
        if not data.get("goals") or "reference" not in data:
            return False
        ref = cv2.imdecode(np.frombuffer(base64.b64decode(data["reference"]), np.uint8),
                           cv2.IMREAD_GRAYSCALE)
        self.stab.set_reference(small_gray=ref)
        resp, _ = self.stab.match(first_frame)
        if resp < 0.20:
            self.stab.set_reference(first_frame)
            print(f"  [корзина] сохранённая корзина не подходит к этой сцене "
                  f"(совпадение {resp:.2f}) - ищу заново")
            return False
        self.goals = [Goal.from_json(g, self.w, self.h) for g in data["goals"]]
        if not self.stab.follow:
            _, shift = self.stab.match(first_frame)
            for g in self.goals + self.human:
                g.x0 += shift[0]; g.x1 += shift[0]; g.y0 += shift[1]; g.y1 += shift[1]
            self.stab.set_reference(first_frame)
        print(f"  [корзина] взята из файла (совпадение сцены {resp:.2f})")
        return True


def draw(vis, c: BallCounter, d, fps_real, elapsed, status=""):
    r = c.model.r
    flash = c.fi <= c.flash
    for z in c.human:
        if not c.visible(z):
            continue
        p0 = (int(z.x0 + d[0]), int(z.y0 + d[1]))
        p1 = (int(z.x1 + d[0]), int(z.y1 + d[1]))
        cv2.rectangle(vis, p0, p1, (150, 150, 150), 2)
        cv2.putText(vis, "HUMAN - not counted", (p0[0] + 3, max(12, p0[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    if c.tag_seen is not None and c.fi - c.tag_seen[0] < c.fps:
        cv2.polylines(vis, [np.int32(c.tag_seen[1])], True, (255, 160, 0), 2)
    for gi, g in enumerate(c.goals):
        if not c.visible(g):
            continue
        p0 = (int(g.x0 + d[0]), int(g.y0 + d[1]))
        p1 = (int(g.x1 + d[0]), int(g.y1 + d[1]))
        seen = g.anchor is None or g.anchor.seen
        col = (0, 255, 0) if flash else ((0, 255, 255) if seen else (0, 140, 255))
        cv2.rectangle(vis, p0, p1, col, 3 if flash else 2)
        if len(c.goals) > 1 or not seen:
            cv2.putText(vis, f"{gi}: {g.score}" + ("" if seen else " ?"), (p0[0] + 4, p0[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for tr in c.tracker.tracks:
        if not getattr(tr, "seen", False):
            continue
        col = (0, 255, 0) if tr.counted >= 0 else ((255, 0, 255) if tr.flight else (160, 160, 160))
        cv2.circle(vis, (int(tr.x + d[0]), int(tr.y + d[1])), int(1.6 * r) + 2, col, 2)

    m, sec = divmod(int(elapsed), 60)
    lines = [(f"SCORE: {c.total}", 1.0, (0, 165, 255), 2),
             (f"TIME: {m}:{sec:02d}", 0.7, (255, 255, 255), 2),
             (f"fps {fps_real:.0f}", 0.45, (200, 200, 200), 1)]
    if c.tag_ref is not None:
        seen = c.tag_seen is not None and c.fi - c.tag_seen[0] < c.fps
        lines.append((f"AprilTag id={c.tag_ref['id']}" + ("" if seen else " (not visible)"),
                      0.45, (255, 160, 0) if seen else (0, 0, 255), 1))
    if not c.goals:
        n = len(c.landings)
        lines.append((f"searching basket: {n}/{c.s.learn_shots} shots  (C - set by hand)",
                      0.5, (0, 255, 255), 1))
    if c.goals and not all(c.visible(g) for g in c.goals):
        lines.append(("basket not visible - searching", 0.5, (0, 140, 255), 1))
    if status:
        lines.append((status, 0.5, (0, 255, 255), 1))
    wbox = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, th)[0][0] for t, sc, _, th in lines) + 16
    hbox = 10 + sum(int(32 * sc) + 6 for _, sc, _, _ in lines)
    cv2.rectangle(vis, (0, 0), (wbox, hbox), (0, 0, 0), -1)
    y = 6
    for t, sc, col, th in lines:
        y += int(32 * sc)
        cv2.putText(vis, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
        y += 6
    return vis


MAX_MANUAL = 4

GROW_KEYS = {2490368, 65362, 63232, ord("]"), 1098, 250}
SHRINK_KEYS = {2621440, 65364, 63233, ord("["), 1093, 245}


def window_available():
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        return True
    except cv2.error:
        return False


WEB_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FGC ball counter</title>
<style>
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center}
 #score{font-size:22vw;font-weight:800;color:#ffa500;line-height:1}
 #info{font-size:5vw;color:#ccc;margin:4px 0 8px}
 img{width:100%;max-width:900px;display:block;margin:0 auto}
 .b{display:inline-block;margin:10px 6px;padding:14px 22px;font-size:6vw;border-radius:12px;
    border:0;background:#333;color:#fff}
 #status{color:#ff0;font-size:4vw;min-height:5vw}
</style></head><body>
<div id="score">0</div><div id="info">...</div><div id="status"></div>
<img src="/stream">
<button class="b" onclick="act('minus')">-1</button>
<button class="b" onclick="act('reset')">СБРОС</button>
<button class="b" onclick="act('plus')">+1</button>
<script>
function act(a){fetch('/'+a,{method:'POST'})}
async function tick(){try{const r=await fetch('/state');const s=await r.json();
 document.getElementById('score').textContent=s.score;
 document.getElementById('info').textContent='время '+s.time+'   fps '+s.fps;
 document.getElementById('status').textContent=s.status||'';}catch(e){}
 setTimeout(tick,300)}
tick();
</script></body></html>"""


def lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class WebView:
    def __init__(self, port):
        import http.server
        import threading
        self.jpeg = None
        self.state = {"score": 0, "time": "0:00", "fps": 0, "status": ""}
        self.commands = []
        self.lock = threading.Lock()
        view = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/state":
                    with view.lock:
                        body = json.dumps(view.state).encode("utf-8")
                    self._send(body, "application/json")
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            with view.lock:
                                jpg = view.jpeg
                            if jpg is not None and jpg is not last:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                                 b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                                self.wfile.write(jpg + b"\r\n")
                                last = jpg
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        return
                else:
                    self._send(WEB_PAGE.encode("utf-8"), "text/html; charset=utf-8")

            def do_POST(self):
                cmd = self.path.strip("/")
                if cmd in ("reset", "plus", "minus"):
                    with view.lock:
                        view.commands.append(cmd)
                self._send(b"ok", "text/plain")

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://{lan_ip()}:{port}"

    def publish(self, vis, state, max_side=720):
        h, w = vis.shape[:2]
        k = min(1.0, max_side / float(max(h, w)))
        small = cv2.resize(vis, (int(w * k), int(h * k))) if k < 1 else vis
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self.lock:
            if ok:
                self.jpeg = buf.tobytes()
            self.state = state

    def take_commands(self):
        with self.lock:
            cmds, self.commands = self.commands, []
        return cmds

    def close(self):
        self.server.shutdown()


def load_config(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"  {path} повреждён - начинаю с чистого")
    return {}


def save_config(path, cfg):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def make_settings(cfg, overrides):
    s = Settings()
    types = {f.name: f.type for f in fields(Settings)}
    items = list((cfg.get("settings") or {}).items())
    for o in overrides:
        k, _, v = o.partition("=")
        items.append((k.strip(), v.strip()))
    for k, v in items:
        if k not in types:
            print(f"  неизвестная настройка: {k}")
            continue
        t = types[k]
        if t in (bool, "bool"):
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on", "da")
        elif t in (int, "int"):
            v = int(float(v))
        elif t in (float, "float"):
            v = float(v)
        setattr(s, k, v)
    return s


def run(args):
    _quiet_opencv()
    cfg = load_config(args.config)
    s = make_settings(cfg, args.set)
    if args.goals:
        s.n_goals = args.goals
    if args.learn_shots:
        s.learn_shots = args.learn_shots

    src = Source(args.source, args)
    ok, frame = src.read()
    if not ok:
        sys.exit("Источник не отдаёт кадры")
    H0, W0 = frame.shape[:2]
    scale = min(1.0, s.proc_max_side / float(max(H0, W0)))

    def prep(f):
        return cv2.resize(f, (int(W0 * scale), int(H0 * scale))) if scale < 1.0 else f

    frame = prep(frame)
    print(f"Источник: {src.name}  {W0}x{H0}  {src.fps:.0f} fps"
          f"{f'  (обработка {frame.shape[1]}x{frame.shape[0]})' if scale < 1 else ''}")

    counter = BallCounter(s, src.fps, frame.shape)
    counter.last_frame = frame
    counter.stab.set_reference(frame)
    loaded = False
    if not args.relearn:
        loaded = counter.load(cfg, frame)
    elif "ball" in cfg and not args.recolor:
        counter.model.load(cfg["ball"], frame.shape[0])
    if args.recolor:
        counter.model = BallModel(s, frame.shape[0])
    if args.roi:
        x0, y0, x1, y1 = [float(v) * scale for v in args.roi.split(",")]
        counter.stab.set_reference(frame)
        counter.set_goals([Goal(x0, y0, x1, y1, source="manual")])
    if not loaded and not args.roi:
        print("  [корзина] жду бросков, чтобы найти корзину...")

    show = not args.no_window and window_available()
    ui = {"clicks": [], "mode": None, "mouse": None}
    if show:
        def on_mouse(event, x, y, flags, param):
            ui["mouse"] = (x, y)
            if event == cv2.EVENT_LBUTTONDOWN and ui["mode"] in ("roi", "human"):
                ui["clicks"].append((x, y))
        cv2.setMouseCallback(WINDOW, on_mouse)

    web = None
    if s.web_port and not args.no_web:
        try:
            web = WebView(s.web_port)
            print(f"  ТЕЛЕФОН: откройте {web.url}  (телефон в той же Wi-Fi сети, что и компьютер)")
        except OSError as e:
            print(f"  просмотр с телефона не запустился (порт {s.web_port} занят?): {e}")

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src.fps,
                                 (frame.shape[1], frame.shape[0]))

    t_start = time.time()
    t_prev = time.time()
    fps_real = src.fps
    paused = False
    first = True
    dark_n = 0
    while True:
        if not first and not paused:
            ok, raw = src.read()
            if not ok:
                if src.live:
                    print("  камера пропала - переподключаюсь...")
                    time.sleep(0.5)
                    if src.cam_index is not None:
                        cap = open_camera(src.cam_index, args.width, args.height, args.fps)
                        if cap is not None:
                            src.cap = cap
                            continue
                break
            frame = prep(raw)
        first = False
        t = (time.time() - t_start) if src.live else counter.fi / src.fps
        if not paused:
            d = counter.process(frame, t)
        dark_n = dark_n + 1 if frame_brightness(frame) < 25 else 0
        if dark_n == int(src.fps) and src.live:
            print(DARK_HINT)

        now = time.time()
        if now > t_prev:
            fps_real = 0.9 * fps_real + 0.1 / (now - t_prev)
        t_prev = now

        status = ""
        if dark_n >= int(src.fps):
            status = "CAMERA IMAGE IS BLACK - see console"
        elif ui["mode"] == "roi":
            status = ("ROBOT basket: click TOP-LEFT corner" if not ui["clicks"]
                      else "ROBOT basket: click BOTTOM-RIGHT corner") + "  (Esc - cancel)"
        elif ui["mode"] == "human":
            status = ("HUMAN basket (not counted): click TOP-LEFT" if not ui["clicks"]
                      else "HUMAN basket: click BOTTOM-RIGHT") + "  (Esc - cancel)"
        if show or writer is not None or web is not None:
            vis = frame.copy()
            if ui["mode"] and ui["mouse"]:
                mx, my = ui["mouse"]
                cv2.line(vis, (mx, 0), (mx, vis.shape[0]), (0, 0, 255), 1)
                cv2.line(vis, (0, my), (vis.shape[1], my), (0, 0, 255), 1)
                if ui["clicks"]:
                    cv2.rectangle(vis, ui["clicks"][0], (mx, my), (0, 0, 255), 2)
            vis = draw(vis, counter, d, fps_real, t, status)
            if writer is not None:
                writer.write(vis)
            if show:
                cv2.imshow(WINDOW, vis)
            if web is not None:
                m, sec = divmod(int(t), 60)
                web.publish(vis, {"score": counter.total, "time": f"{m}:{sec:02d}",
                                  "fps": int(fps_real), "status": status})
                for cmd in web.take_commands():
                    if cmd == "reset":
                        counter.reset()
                        t_start = time.time()
                        print("  счёт сброшен (с телефона)")
                    elif counter.goals:
                        counter._event(0, +1 if cmd == "plus" else -1, "вручную")
        if src.n_frames and counter.fi % 300 == 0 and not show:
            print(f"  ... {100 * counter.fi / src.n_frames:.0f}%")
        if not show:
            continue

        k = cv2.waitKeyEx(1)
        key = k & 0xFF if 0 <= k < 256 else -1
        if ui["mode"] and len(ui["clicks"]) == 2:
            (ax, ay), (bx, by) = ui["clicks"]
            x0, x1 = sorted((ax - d[0], bx - d[0]))
            y0, y1 = sorted((ay - d[1], by - d[1]))
            if x1 - x0 >= 5 and y1 - y0 >= 5:
                if ui["mode"] == "roi":
                    new = Goal(x0, y0, x1, y1, source="manual")
                    keep = [g for g in counter.goals if g.source in ("manual", "auto_red")
                            and overlap(g, new) < 0.3]
                    if len(keep) >= MAX_MANUAL:
                        keep = keep[1:]
                    counter.set_goals(keep + [new])
                    print(f"  [корзина] рамка корзины робота задана вручную (всего рамок: {len(counter.goals)})")
                else:
                    counter.human.append(Goal(x0, y0, x1, y1, source="human"))
                    print("  [корзина] зона человеческой корзины задана - мячи в ней не считаются")
            ui["mode"], ui["clicks"] = None, []
        if k in GROW_KEYS or k in SHRINK_KEYS:
            if counter.goals:
                g = counter.goals[-1]
                f = 1.05 if k in GROW_KEYS else 1 / 1.05
                cx, cy = (g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2
                g.x0, g.x1 = cx + (g.x0 - cx) * f, cx + (g.x1 - cx) * f
                g.y0, g.y1 = cy + (g.y0 - cy) * f, cy + (g.y1 - cy) * f
                g.source = "manual"
                g.anchor = None
                print(f"  рамка {'больше' if f > 1 else 'меньше'}: x {g.x0:.0f}..{g.x1:.0f}, y {g.y0:.0f}..{g.y1:.0f}")
        elif key == 27 and ui["mode"]:
            ui["mode"], ui["clicks"] = None, []
        elif key in (27, ord("q"), ord("Q")):
            break
        elif key in (ord("r"), ord("R")):
            counter.reset()
            t_start = time.time()
            print("  счёт сброшен")
        elif key in (ord("l"), ord("L")):
            counter.relearn()
        elif key in (ord("c"), ord("C")):
            ui["mode"], ui["clicks"] = "roi", []
        elif key in (ord("h"), ord("H")):
            ui["mode"], ui["clicks"] = "human", []
        elif key in (ord("x"), ord("X")) and counter.human:
            counter.human = []
            print("  зоны человеческой корзины убраны")
        elif key in (ord("z"), ord("Z")) and counter.goals:
            counter.set_goals([])
            print("  корзины робота убраны (C - задать заново, или жду бросков)")
        elif key in (ord("n"), ord("N")):
            if src.next_camera():
                ok, raw = src.read()
                if ok:
                    H0, W0 = raw.shape[:2]
                    scale = min(1.0, s.proc_max_side / float(max(H0, W0)))
                    frame = prep(raw)
                    counter = BallCounter(s, src.fps, frame.shape)
                    counter.last_frame = frame
                    counter.stab.set_reference(frame)
                    if writer is not None:
                        writer.release()
                        writer = None
                    t_start = time.time()
                    first = True
                print(f"  переключился: {src.name} - ищу корзину")
        elif key in (ord("+"), ord("=")) and counter.goals:
            counter._event(0, +1, "вручную")
        elif key in (ord("-"), ord("_")) and counter.goals:
            counter._event(0, -1, "вручную")
        elif key == ord(" "):
            paused = not paused

    if web is not None:
        web.close()
    src.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    if counter.goals or counter.model.fitted:
        data = counter.to_json()
        data["settings"] = cfg.get("settings", {})
        if not counter.goals and cfg.get("goals"):
            data["goals"], data["reference"] = cfg["goals"], cfg.get("reference")
            if cfg.get("look"):
                data["look"] = cfg["look"]
        save_config(args.config, data)
        print(f"  настройки сохранены -> {args.config}")

    print("\n=== РЕЗУЛЬТАТ ===")
    for gi, g in enumerate(counter.goals):
        print(f"  корзина {gi}: {g.score}")
    print(f"  ВСЕГО: {counter.total}")

    if args.report:
        os.makedirs(args.report, exist_ok=True)
        with open(os.path.join(args.report, "events.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["time_s", "frame", "goal", "delta", "goal_score", "total", "reason"])
            w.writeheader()
            w.writerows(counter.events)
        with open(os.path.join(args.report, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"source": src.name, "total": counter.total,
                       "goals": [{"goal": i, "score": g.score,
                                  "bbox": [round(v / scale) for v in (g.x0, g.y0, g.x1, g.y1)]}
                                 for i, g in enumerate(counter.goals)]},
                      f, ensure_ascii=False, indent=2)
        print(f"  отчёт -> {args.report}/")
    return counter.total


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FGC 2026 - счётчик мячей в корзине")
    p.add_argument("words", nargs="*", help="видео / ссылка / номер камеры "
                   "(можно со старыми словами calibrate|count|auto)")
    p.add_argument("--camera", type=int, help="номер камеры")
    p.add_argument("--list-cameras", action="store_true", help="показать доступные камеры")
    p.add_argument("--config", "--calib", default=DEFAULT_CONFIG, help="файл настроек (goals.json)")
    p.add_argument("--relearn", action="store_true", help="найти корзину заново")
    p.add_argument("--recolor", action="store_true", help="подобрать цвет мяча заново")
    p.add_argument("--roi", help="корзина вручную: x0,y0,x1,y1 (в пикселях кадра)")
    p.add_argument("--goals", type=int, help="сколько корзин искать (по умолчанию 1)")
    p.add_argument("--learn-shots", type=int, help="сколько бросков нужно, чтобы найти корзину")
    p.add_argument("--width", type=int, help="ширина кадра камеры")
    p.add_argument("--height", type=int, help="высота кадра камеры")
    p.add_argument("--fps", type=int, help="fps камеры (больше = точнее на быстрых бросках)")
    p.add_argument("--no-window", "--no-video", action="store_true", help="без окна")
    p.add_argument("--no-web", action="store_true", help="не запускать страницу для телефона")
    p.add_argument("--save", help="записать видео с разметкой")
    p.add_argument("--report", "--out", help="папка для events.csv и summary.json")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="изменить любую настройку, например --set learn_shots=2")
    a = p.parse_args(argv)

    words = list(a.words)
    if words and words[0] in ("calibrate", "count", "auto"):
        mode = words.pop(0)
        if mode in ("calibrate", "auto"):
            a.relearn = True
        if mode == "calibrate":
            a.no_window = True
    a.source = words[0] if words else (a.camera if a.camera is not None else None)
    return a


def main():
    a = parse_args()
    if a.list_cameras:
        cams = list_cameras()
        if not cams:
            print("Камеры не найдены")
        for i, w, h in cams:
            print(f"  камера #{i}: {w}x{h}")
        return
    run(a)


if __name__ == "__main__":
    main()
