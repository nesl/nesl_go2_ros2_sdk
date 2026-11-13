"""Camera trail: where the dog has been on the lab map, with what its camera saw there.

While the console is lined up with the saved map, a 640 px camera frame is kept every time the
dog moves 0.3 m or turns 20 degrees, with its map pose. Later a person can click a spot on the map
and see the views taken near it, to mark places after the fact.

Stored in /maps/<name>_trail/<YYYYMMDD>/: one <ms>.jpg per frame and index.jsonl with the poses.
Days older than KEEP_DAYS are deleted at startup.
"""
import json
import math
import os
import re
import shutil
import threading
import time

STEP_M, STEP_RAD = 0.3, math.radians(20)   # keep a frame after moving or turning this much
MAX_FRAME_AGE = 0.3                         # only keep frames this fresh (s)
MAX_TURN = 0.3                              # and not while turning fast (rad/s): blurry, pose lags
KEEP_DAYS = 14
FOV_HALF_DEG = 40                            # the front camera sees about 40 degrees to each side
HOST_UID = 1000


class Trail:
    def __init__(self, maps_dir, name):
        self.root = os.path.join(maps_dir, f"{name}_trail")
        self.lock = threading.Lock()
        self.points = []        # [t, x, y, yaw, "YYYYMMDD/<ms>.jpg"], oldest first
        self.last = None        # (x, y, yaw) of the last kept frame
        self.last_t = None      # and its camera time
        os.makedirs(self.root, exist_ok=True)
        os.chown(self.root, HOST_UID, HOST_UID)
        self.prune()
        self.load()

    def prune(self):
        cutoff = time.strftime("%Y%m%d", time.localtime(time.time() - KEEP_DAYS * 86400))
        for d in os.listdir(self.root):
            if re.fullmatch(r"\d{8}", d) and d < cutoff:
                shutil.rmtree(os.path.join(self.root, d), ignore_errors=True)

    def load(self):
        for d in sorted(os.listdir(self.root)):
            try:
                with open(os.path.join(self.root, d, "index.jsonl")) as f:
                    for line in f:
                        try:
                            p = json.loads(line)
                            self.points.append([p["t"], p["x"], p["y"], p["yaw"], f"{d}/{p['f']}"])
                        except (ValueError, KeyError):
                            pass
            except OSError:
                pass

    def offer(self, pose, jpeg_bytes, frame_t, turn_rate):
        """Called a few times a second with the map pose and the latest 640 px frame."""
        if pose is None or jpeg_bytes is None or time.time() - frame_t > MAX_FRAME_AGE or abs(turn_rate) > MAX_TURN:
            return
        if frame_t == self.last_t:      # same frame as last time (camera stalled)
            return
        x, y, yaw = pose["x"], pose["y"], pose["yaw"]
        if self.last is not None:
            lx, ly, lyaw = self.last
            dyaw = abs((yaw - lyaw + math.pi) % (2 * math.pi) - math.pi)
            if math.hypot(x - lx, y - ly) < STEP_M and dyaw < STEP_RAD:
                return
        self.last, self.last_t = (x, y, yaw), frame_t
        day = time.strftime("%Y%m%d", time.localtime(frame_t))
        d = os.path.join(self.root, day)
        fresh_day = not os.path.isdir(d)
        os.makedirs(d, exist_ok=True)
        name = f"{int(frame_t * 1000)}.jpg"
        with open(os.path.join(d, name), "wb") as f:
            f.write(jpeg_bytes)
        with open(os.path.join(d, "index.jsonl"), "a") as f:
            f.write(json.dumps({"t": round(frame_t, 3), "x": x, "y": y, "yaw": yaw, "f": name}) + "\n")
        for p in (d, os.path.join(d, name), os.path.join(d, "index.jsonl")) if fresh_day else (os.path.join(d, name),):
            os.chown(p, HOST_UID, HOST_UID)
        with self.lock:
            self.points.append([round(frame_t, 3), x, y, yaw, f"{day}/{name}"])

    def near(self, x, y, radius=3.0, n=12):
        """Frames taken within radius of (x, y): first the ones whose camera faced the spot, nearest
        first, then the rest. At most one per 0.15 m spot and 30 degrees of heading."""
        with self.lock:
            pts = [p for p in self.points if math.hypot(p[1] - x, p[2] - y) <= radius]
        rows = []
        for t, px, py, yaw, f in pts:
            dist = math.hypot(px - x, py - y)
            # where the spot is in the picture: degrees left (+) or right (-) of straight ahead
            bearing = math.degrees((math.atan2(y - py, x - px) - yaw + math.pi) % (2 * math.pi) - math.pi)
            facing = dist < 0.3 or abs(bearing) <= FOV_HALF_DEG
            rows.append((not facing, dist, -t, t, px, py, yaw, f, bearing))
        rows.sort()
        out, taken = [], set()
        for away, dist, _, t, px, py, yaw, f, bearing in rows:
            key = (round(px / 0.15), round(py / 0.15), round(yaw / math.radians(30)) % 12)
            if key in taken:
                continue
            taken.add(key)
            out.append({"t": t, "x": round(px, 2), "y": round(py, 2), "yaw_deg": round(math.degrees(yaw), 1),
                        "dist": round(dist, 2), "bearing_deg": round(bearing), "facing": not away,
                        "url": f"/trail/{f}"})
            if len(out) >= n:
                break
        return out

    def coverage(self, cell=0.25):
        """Spots (x, y) where frames exist, one per cell, for drawing on the map."""
        with self.lock:
            cells = {(round(p[1] / cell), round(p[2] / cell)) for p in self.points}
        return [[round(i * cell, 2), round(j * cell, 2)] for i, j in cells]

    def path(self, rel):
        """Local file for a /trail/<day>/<ms>.jpg URL, or None."""
        if not re.fullmatch(r"\d{8}/\d+\.jpg", rel):
            return None
        p = os.path.join(self.root, rel)
        return p if os.path.isfile(p) else None
