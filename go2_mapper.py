"""Lab map built straight from the dog's lidar clouds, which arrive already placed in the dog's own
world frame (odom). The dog's onboard estimate is clean, so no scan matching while mapping.

The odom frame restarts whenever the dog reboots, so each session needs one transform from odom to
the saved map. `align()` finds it by matching what the lidar sees now against the saved map
(rotation by brute force, translation by FFT correlation). Places and map_pose are in map frame.
"""
import json
import math
import os
import shutil
import threading
import time

import cv2
import numpy as np

RES = 0.05                 # m per cell
SIZE = 1000                # cells per side: a 50 m square, centred on where mapping started
MIN_H, MAX_H = 0.20, 0.60  # obstacle heights above the floor that go on the map: what the dog can bump into (m).
                           # The lidar's voxels are ~10 cm tall, so floor points scatter up to ~15 cm.
SCORE_CAP = 5              # each cell's score: +1 per obstacle sighting, -1 per clear sighting, kept within ±5,
OCC_SCORE = 2              # so a few clear looks erase a stale obstacle; occupied at a score of 2 or more
SAME_SESSION_M, SAME_SESSION_RAD = 1.0, 0.5   # dog this close to where it was: same odom frame, keep the alignment
ALIGN_OK = 0.5             # (points on saved walls - points on saved open floor) / points, at the best fit
ALIGN_MARGIN = 0.15        # best fit must beat the best clearly different rotation by this much
ALIGN_RETRY_S = 20.0       # wait this long before trying again after a failed line-up
ALIGN_MAX_PTS = 2500
TRACK_RANGE_M, TRACK_STEP_CELLS = 0.20, 1    # drift correction: search ±20 cm in 5 cm steps
TRACK_DEG, TRACK_DEG_STEP = 2.0, 0.5          # and ±2 degrees in half-degree steps, around the robot
TRACK_SIGMA = 0.08         # m: how sharply a lidar point must sit on a mapped wall to count
TRACK_MIN_PTS = 80         # need this many obstacle points in a view to correct from it
TRACK_MIN_FIT = 0.35       # mean wall likelihood at the best fit
TRACK_MIN_GAIN = 0.03      # best fit must beat "no correction" by this much
TRACK_GAIN = 0.6           # apply this fraction of each correction (smooths noise)
TRACK_MAX_PTS = 700
MISMATCH_KNOWN = 0.6       # a view whose points mostly fall on mapped area (this fraction)...
MISMATCH_FIT = 0.25        # ...but sit this poorly on its walls even after correcting is off: don't map it
HOST_UID = 1000


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mapper:
    def __init__(self, maps_dir, name):
        self.dir, self.name = maps_dir, name
        self.lock = threading.Lock()
        self.score = np.zeros((SIZE, SIZE), np.int8)
        self.seen = np.zeros((SIZE, SIZE), bool)
        self.origin = (-SIZE * RES / 2, -SIZE * RES / 2)   # map-frame xy of cell (0, 0)
        self.T = None              # (tx, ty, th): map = R(th) * odom + t; None until aligned
        self.mapping = False
        self.status = "no map"
        self.align_info = None
        self.recent = []           # recent obstacle clouds in odom frame, for aligning
        self.version = 0
        self.saved_at = None
        self._aligning = False
        self._last_try = 0.0
        self.resume_mapping = False
        self.pending = None        # (T, odom) stored with the map, reused if the dog hasn't rebooted
        self.field = None          # (likelihood image, row0, col0, key) for drift correction
        self.track_info = None
        self.tracking = True       # correct drift by matching each lidar view to the map
        self.skipped = 0           # lidar views left off the map because they didn't match it
        if self.load_unsaved():
            self.status = "restored the unsaved map from before the console restarted; lining up with it"
        elif self.load():
            self.status = "saved map loaded; lining up with it"
            self.load_session()
        else:
            self.status = "no saved map: press Start mapping"

    # ---------- frames ----------
    def to_map(self, x, y, yaw=0.0):
        tx, ty, th = self.T
        c, s = math.cos(th), math.sin(th)
        return tx + c * x - s * y, ty + s * x + c * y, wrap(yaw + th)

    def pose(self, odom):
        if self.T is None or odom is None:
            return None
        x, y, yaw = self.to_map(odom["pos"][0], odom["pos"][1], odom["yaw"])
        return {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)}

    def cells(self, xy):
        """map-frame points (N,2) -> unique in-bounds cell indices (rows, cols), row 0 at the bottom."""
        ij = np.floor((xy - np.array(self.origin)) / RES).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < SIZE) & (ij[:, 1] >= 0) & (ij[:, 1] < SIZE)
        ij = np.unique(ij[ok], axis=0)
        return ij[:, 1], ij[:, 0]

    # ---------- updates from the lidar ----------
    def add_cloud(self, pts, floor, robot=None):
        """pts: (N,3) odom-frame points near the robot; floor: floor height estimate;
        robot: (x, y) of the robot in the odom frame."""
        h = pts[:, 2] - floor
        obs = pts[(h > MIN_H) & (h < MAX_H), :2]
        flat = pts[h <= MIN_H, :2]
        self.recent = (self.recent + [obs[::2].copy()])[-4:]
        ok = True
        if self.T is not None and robot is not None and self.tracking and not self._aligning:
            ok = self.track(obs, robot)
        if not self.mapping or self.T is None:
            return
        if not ok:
            self.skipped += 1
            self.status = (f"extending the saved map; skipped {self.skipped} lidar views that don't match it "
                           "(the dog's position on the map may be off: drive back to a well-mapped spot)")
            return
        tx, ty, th = self.T
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        # one vote per cell per cloud: obstacle if anything in the height band, else open floor
        orr, oc = self.cells(obs @ R.T + (tx, ty))
        fr, fc = self.cells(flat @ R.T + (tx, ty))
        keep = ~np.isin(fr * SIZE + fc, orr * SIZE + oc)
        fr, fc = fr[keep], fc[keep]
        with self.lock:
            self.score[orr, oc] = np.minimum(self.score[orr, oc] + 1, SCORE_CAP)
            self.score[fr, fc] = np.maximum(self.score[fr, fc] - 1, -SCORE_CAP)
            self.seen[orr, oc] = True
            self.seen[fr, fc] = True
            self.version += 1

    def cell_state(self, x, y):
        """'obstacle', 'open' or 'unseen' at map point (x, y)."""
        c, r = int(math.floor((x - self.origin[0]) / RES)), int(math.floor((y - self.origin[1]) / RES))
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            return "unseen"
        with self.lock:
            score, seen = int(self.score[r, c]), bool(self.seen[r, c])
        return "obstacle" if score >= OCC_SCORE else "open" if seen else "unseen"

    # ---------- the map as an image ----------
    def grid(self):
        """uint8 image (row 0 at the bottom): 0 occupied, 254 free, 205 unknown."""
        with self.lock:
            score, seen = self.score.copy(), self.seen.copy()
        img = np.full((SIZE, SIZE), 205, np.uint8)
        occ = score >= OCC_SCORE
        img[seen] = 254
        img[occ] = 0
        return img

    def crop(self, img, margin=20):
        known = np.argwhere(img != 205)
        if len(known) == 0:
            return None
        (r0, c0), (r1, c1) = known.min(0) - margin, known.max(0) + margin + 1
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, SIZE), min(c1, SIZE)
        return img[r0:r1, c0:c1], (self.origin[0] + c0 * RES, self.origin[1] + r0 * RES)

    def png(self):
        """(png bytes, meta) of the known part of the map, north up; None if empty."""
        cropped = self.crop(self.grid())
        if cropped is None:
            return None, None
        img, (ox, oy) = cropped
        ok, buf = cv2.imencode(".png", img[::-1])
        seen = int((img != 205).sum())
        meta = {"resolution": RES, "origin": [round(ox, 3), round(oy, 3)], "width": int(img.shape[1]),
                "height": int(img.shape[0]), "t": round(time.time(), 2), "mapped_m2": round(seen * RES * RES, 1),
                "mode": self.mode(), "status": self.status, "align": self.align_info, "saved_at": self.saved_at,
                "track": self.track_info}
        return buf.tobytes(), meta

    def mode(self):
        if self.mapping:
            return "mapping"
        return "localized" if self.T is not None else "not_aligned"

    # ---------- commands ----------
    def start_mapping(self, odom, fresh):
        with self.lock:
            if fresh:
                self.score[:] = 0
                self.seen[:] = False
                if self.T is None:
                    self.T = (0.0, 0.0, 0.0)   # new map frame = this session's odom frame
                # else keep the current frame, so saved places stay where they were
                self.status = "mapping a new map"
            elif self.T is None:
                return "Not lined up with the saved map yet, so it can't be extended. Start a new map instead."
            else:
                self.status = "extending the saved map"
            self.skipped = 0
            self.mapping = True
        return None

    def clear(self):
        """Erase the map in memory. The saved file stays until the next save."""
        with self.lock:
            self.score[:] = 0
            self.seen[:] = False
            self.version += 1
            if self.T is None:
                self.T = (0.0, 0.0, 0.0)
        self.status = "cleared; mapping from scratch" if self.mapping else "cleared (the saved map is unchanged until you save)"
        return None

    def stop_mapping(self):
        self.mapping = False
        self.resume_mapping = False
        self.drop_unsaved()
        self.status = "mapping stopped (not saved)"

    def backup(self, keep=20):
        """Copy the current saved map to history/<name>_<saved time>.* before it is overwritten."""
        path = os.path.join(self.dir, self.name)
        if not os.path.exists(path + ".grid.npz"):
            return
        hist = os.path.join(self.dir, "history")
        os.makedirs(hist, exist_ok=True)
        os.chown(hist, HOST_UID, HOST_UID)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(path + ".grid.npz")))
        for ext in (".grid.npz", ".pgm", ".yaml"):
            if os.path.exists(path + ext):
                dst = os.path.join(hist, f"{self.name}_{stamp}{ext}")
                shutil.copy2(path + ext, dst)
                os.chown(dst, HOST_UID, HOST_UID)
        old = sorted(f for f in os.listdir(hist) if f.startswith(self.name + "_") and f.endswith(".grid.npz"))
        for f in old[:-keep]:
            for ext in (".grid.npz", ".pgm", ".yaml"):
                try:
                    os.remove(os.path.join(hist, f[:-len(".grid.npz")] + ext))
                except FileNotFoundError:
                    pass

    def revert(self):
        """Throw away unsaved changes: stop mapping and reload the saved map. The alignment is kept,
        since the saved map is in the same frame."""
        self.mapping = False
        self.resume_mapping = False
        self.drop_unsaved()
        path = os.path.join(self.dir, self.name + ".grid.npz")
        if not os.path.exists(path):
            return "There is no saved map to go back to."
        d = np.load(path)
        with self.lock:
            self.read_grid(d)
            self.origin = tuple(d["origin"])
            self.version += 1
        self.field = None
        self.saved_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
        self.status = f"back to the saved map from {self.saved_at}"
        return None

    def save(self, odom=None):
        img = self.grid()
        path = os.path.join(self.dir, self.name)
        self.backup()
        with self.lock:
            np.savez_compressed(path + ".grid.npz", score=self.score, seen=self.seen, origin=np.array(self.origin),
                                **self.frame_info(odom))
        cropped = self.crop(img)
        if cropped is not None:   # standard map files too, for viewing and for Nav2
            c, (ox, oy) = cropped
            cv2.imwrite(path + ".pgm", c[::-1])
            with open(path + ".yaml", "w") as f:
                f.write(f"image: {self.name}.pgm\nmode: trinary\nresolution: {RES}\norigin: [{ox:.3f}, {oy:.3f}, 0]\n"
                        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
        for ext in (".grid.npz", ".pgm", ".yaml"):
            if os.path.exists(path + ext):
                os.chown(path + ext, HOST_UID, HOST_UID)
        self.mapping = False
        self.drop_unsaved()
        self.saved_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.status = f"saved {self.saved_at}"
        return None

    def read_grid(self, d):
        if "score" in d:
            self.score[:], self.seen[:] = d["score"], d["seen"]
        else:   # older files kept raw counts
            self.score[:] = np.clip(d["hits"].astype(np.int32) - d["miss"], -SCORE_CAP, SCORE_CAP)
            self.seen[:] = (d["hits"].astype(np.int32) + d["miss"]) > 0

    def unsaved_path(self):
        return os.path.join(self.dir, self.name + ".unsaved.npz")

    def frame_info(self, odom):
        """The alignment plus where the dog was, so a restart without a dog reboot can reuse it."""
        if self.T is None or odom is None:
            return {}
        return {"T": np.array(self.T), "odom": np.array([odom["pos"][0], odom["pos"][1], odom["yaw"]])}

    # ---------- remembering the line-up across console restarts ----------
    def session_path(self):
        return os.path.join(self.dir, self.name + ".session.json")

    def save_session(self, odom):
        """Remember the current line-up, so a console restart (same dog session) doesn't lose it."""
        if self.T is None or odom is None or self.mapping:
            return
        tmp = self.session_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"T": list(self.T), "odom": [odom["pos"][0], odom["pos"][1], odom["yaw"]],
                       "saved_at": self.saved_at, "t": time.time()}, f)
        os.chown(tmp, HOST_UID, HOST_UID)
        os.replace(tmp, self.session_path())

    def load_session(self):
        try:
            with open(self.session_path()) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return
        # only for the same saved map and recent; try_align then checks the dog hasn't rebooted
        if d.get("saved_at") == self.saved_at and time.time() - d.get("t", 0) < 12 * 3600:
            self.pending = (tuple(d["T"]), tuple(d["odom"]))

    # ---------- drift correction ----------
    def _wall_field(self):
        """Likelihood that a point sits on a mapped wall, per cell (cached; rebuilt every 3 s while mapping)."""
        key = self.version if not self.mapping else int(time.monotonic() / 3)
        if self.field is not None and self.field[3] == key:
            return self.field
        with self.lock:
            occ = self.score >= OCC_SCORE
        rows, cols = np.nonzero(occ)
        if len(rows) < 50:
            self.field = (None, 0, 0, key)
            return self.field
        m = 20
        r0, c0 = max(rows.min() - m, 0), max(cols.min() - m, 0)
        r1, c1 = min(rows.max() + m, SIZE), min(cols.max() + m, SIZE)
        sub = occ[r0:r1, c0:c1]
        d = cv2.distanceTransform((~sub).astype(np.uint8), cv2.DIST_L2, 5) * RES
        self.field = (np.exp(-d * d / (2 * TRACK_SIGMA ** 2)).astype(np.float32), r0, c0, key)
        return self.field

    def track(self, obs, robot):
        """Nudge T so this lidar view's obstacles sit on the map's walls (odometry drifts, the walls don't).
        Returns False if the view clearly doesn't match the map where it lands (so it shouldn't be mapped)."""
        L, r0, c0, _ = self._wall_field()
        if L is None or len(obs) < TRACK_MIN_PTS:
            return True
        pts = np.unique(np.floor(obs / RES), axis=0) * RES + RES / 2
        if len(pts) < TRACK_MIN_PTS:
            return True
        if len(pts) > TRACK_MAX_PTS:
            pts = pts[np.random.default_rng().choice(len(pts), TRACK_MAX_PTS, replace=False)]
        T = self.T
        tx, ty, th = T
        c, s = math.cos(th), math.sin(th)
        q = pts @ np.array([[c, s], [-s, c]]) + (tx, ty)              # map frame
        rx, ry = tx + c * robot[0] - s * robot[1], ty + s * robot[0] + c * robot[1]
        n = int(round(TRACK_RANGE_M / RES / TRACK_STEP_CELLS))
        offs = np.arange(-n, n + 1) * TRACK_STEP_CELLS
        dr, dc = [a.ravel() for a in np.meshgrid(offs, offs, indexing="ij")]
        H, W = L.shape
        best, zero = None, None
        for deg in np.arange(-TRACK_DEG, TRACK_DEG + 1e-9, TRACK_DEG_STEP):
            a = math.radians(deg)
            ca, sa = math.cos(a), math.sin(a)
            rel = q - (rx, ry)
            qq = rel @ np.array([[ca, sa], [-sa, ca]]) + (rx, ry)
            cc = np.floor((qq[:, 0] - self.origin[0]) / RES).astype(np.int64) - c0
            rr = np.floor((qq[:, 1] - self.origin[1]) / RES).astype(np.int64) - r0
            R = rr[None, :] + dr[:, None]
            C = cc[None, :] + dc[:, None]
            ok = (R >= 0) & (R < H) & (C >= 0) & (C < W)
            vals = np.where(ok, L[np.clip(R, 0, H - 1), np.clip(C, 0, W - 1)], 0.0).mean(axis=1)
            k = int(vals.argmax())
            if best is None or vals[k] > best[0]:
                best = (float(vals[k]), a, dc[k] * RES, dr[k] * RES)
            if abs(deg) < 1e-9:
                zero = float(vals[(dr == 0) & (dc == 0)][0])
        fit, a, dx, dy = best
        # how much of this view lands on area the map already knows (seen cells)
        cc = np.floor((q[:, 0] - self.origin[0]) / RES).astype(np.int64)
        rr = np.floor((q[:, 1] - self.origin[1]) / RES).astype(np.int64)
        inb = (rr >= 0) & (rr < SIZE) & (cc >= 0) & (cc < SIZE)
        known = float(self.seen[rr[inb], cc[inb]].mean()) if inb.any() else 0.0
        info = {"fit": round(fit, 2), "fit_before": round(zero, 2), "points": int(len(pts)),
                "known": round(known, 2), "at": time.strftime("%H:%M:%S")}
        matches = not (known >= MISMATCH_KNOWN and fit < MISMATCH_FIT)
        if fit < TRACK_MIN_FIT or fit - zero < TRACK_MIN_GAIN or (a == 0 and dx == 0 and dy == 0):
            self.track_info = dict(info, applied=False, matches=matches)
            return matches
        a, dx, dy = a * TRACK_GAIN, dx * TRACK_GAIN, dy * TRACK_GAIN
        ca, sa = math.cos(a), math.sin(a)
        # map' = Ra (map - robot) + robot + d, applied to the transform's translation and rotation
        nx = ca * (tx - rx) - sa * (ty - ry) + rx + dx
        ny = sa * (tx - rx) + ca * (ty - ry) + ry + dy
        if self.T is T:           # not replaced meanwhile (line-up, manual frame)
            self.T = (nx, ny, wrap(th + a))
        self.track_info = dict(info, applied=True, matches=matches, dx=round(dx, 3), dy=round(dy, 3),
                               ddeg=round(math.degrees(a), 2))
        return matches

    def take_frame(self, d):
        if "T" in d and "odom" in d:
            self.pending = (tuple(float(v) for v in d["T"]), tuple(float(v) for v in d["odom"]))

    def autosave(self, odom=None):
        """While mapping, keep a copy on disk so a console restart doesn't lose the map."""
        if not self.mapping:
            return
        with self.lock:
            score, seen = self.score.copy(), self.seen.copy()
        tmp = self.unsaved_path() + ".tmp.npz"
        np.savez_compressed(tmp, score=score, seen=seen, origin=np.array(self.origin), **self.frame_info(odom))
        os.chown(tmp, HOST_UID, HOST_UID)
        os.replace(tmp, self.unsaved_path())

    def drop_unsaved(self):
        if os.path.exists(self.unsaved_path()):
            os.remove(self.unsaved_path())

    def load_unsaved(self):
        """After a restart mid-mapping: take the unsaved map back, line up with it, keep mapping."""
        if not os.path.exists(self.unsaved_path()):
            return False
        d = np.load(self.unsaved_path())
        with self.lock:
            self.read_grid(d)
            self.origin = tuple(d["origin"])
        self.take_frame(d)
        self.resume_mapping = True
        return True

    def load(self):
        path = os.path.join(self.dir, self.name + ".grid.npz")
        if not os.path.exists(path):
            return False
        d = np.load(path)
        with self.lock:
            self.read_grid(d)
            self.origin = tuple(d["origin"])
        self.take_frame(d)
        self.saved_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
        return True

    # ---------- lining up a new session with the saved map ----------
    def try_align(self, odom, force=False):
        """Start an alignment in the background if one is needed and the lidar has data."""
        if self._aligning or self.mapping or odom is None or len(self.recent) < 2:
            return
        if self.T is not None and not force:
            return
        if self.pending and not force:
            T, (px, py, pyaw) = self.pending
            self.pending = None
            if (math.hypot(odom["pos"][0] - px, odom["pos"][1] - py) < SAME_SESSION_M
                    and abs(wrap(odom["yaw"] - pyaw)) < SAME_SESSION_RAD):
                self.T = T
                if self.resume_mapping:
                    self.resume_mapping, self.mapping = False, True
                    self.status = "picked up the unsaved map where it left off; still mapping"
                else:
                    self.status = "saved map loaded (same session as when it was saved)"
                return
        if not force and time.monotonic() - self._last_try < ALIGN_RETRY_S:
            return
        if not (self.score >= OCC_SCORE).any():
            return
        self._last_try = time.monotonic()
        self._aligning = True
        self.status = "lining up with the saved map…"
        threading.Thread(target=self._align, args=(odom,), daemon=True).start()

    def _align(self, odom):
        try:
            self.T_found = None
            t0 = time.time()
            pts = np.concatenate(self.recent)          # odom frame
            # one point per 5 cm cell, capped, so the search stays light
            pts = np.unique(np.floor(pts / RES), axis=0) * RES + RES / 2
            if len(pts) > ALIGN_MAX_PTS:
                pts = pts[np.random.default_rng(0).choice(len(pts), ALIGN_MAX_PTS, replace=False)]
            img = self.grid()
            cropped = self.crop(img, margin=60)
            if cropped is None or len(pts) < 50:
                self.status = "not enough lidar data to line up yet"
                return
            sub, (ox, oy) = cropped
            walls = cv2.dilate((sub == 0).astype(np.uint8), np.ones((3, 3), np.uint8))
            free = ((sub == 254) & (walls == 0)).astype(np.float32)
            rel = pts - pts.mean(0)
            pad = int(np.ceil(np.abs(rel).max() * 2 / RES)) + 2     # zero padding so shifts don't wrap around
            H, W = walls.shape[0] + pad, walls.shape[1] + pad
            def ft(a):
                z = np.zeros((H, W), np.float32)
                z[:a.shape[0], :a.shape[1]] = a
                return np.fft.rfft2(z)
            FW, FF = ft(walls.astype(np.float32)), ft(free)
            best = []
            for deg in np.arange(0, 360, 1.0):
                th = math.radians(deg)
                c, s = math.cos(th), math.sin(th)
                q = rel @ np.array([[c, s], [-s, c]])  # rotate by th
                ij = np.unique(np.floor(q / RES).astype(np.int64), axis=0)
                ij -= ij.min(0)
                if ij[:, 0].max() >= W or ij[:, 1].max() >= H:
                    continue
                L = np.zeros((H, W), np.float32)
                L[ij[:, 1], ij[:, 0]] = 1
                FL = np.conj(np.fft.rfft2(L))
                score = (np.fft.irfft2(FW * FL, s=(H, W)) - np.fft.irfft2(FF * FL, s=(H, W))) / len(ij)
                k = int(score.argmax())
                r, cc = divmod(k, W)
                best.append((float(score.flat[k]), deg, r, cc))
            if not best:
                self.status = "could not line up: map too small"
                return
            best.sort(key=lambda b: -b[0])
            score, deg, r, cc = best[0]
            # best fit at a clearly different rotation, to judge ambiguity
            other = next((b[0] for b in best[1:] if abs(wrap(math.radians(b[1] - deg))) > math.radians(10)), 0.0)
            th = math.radians(deg)
            c, s = math.cos(th), math.sin(th)
            q = rel @ np.array([[c, s], [-s, c]])
            qmin = np.floor(q / RES).astype(np.int64).min(0)
            # a rotated point q lands at cell (q/RES - qmin) + (cc, r) of the crop
            shift = np.array([ox, oy]) + (np.array([cc, r]) - qmin) * RES
            mean_map = shift + 0.5 * RES       # map position of the cloud's mean point (rel = 0)
            mx, my = pts.mean(0)
            tx = mean_map[0] - (c * mx - s * my)
            ty = mean_map[1] - (s * mx + c * my)
            self.align_info = {"score": round(score, 2), "runner_up": round(other, 2), "deg": float(deg),
                               "points": int(len(pts)), "secs": round(time.time() - t0, 1), "at": time.strftime("%H:%M:%S")}
            self.T_found = (float(tx), float(ty), th)
            if self.mapping:      # a new map was started meanwhile; its frame wins
                return
            if score >= ALIGN_OK and score - other >= ALIGN_MARGIN:
                self.T = self.T_found
                self.status = f"lined up with the saved map ({score:.0%} fit)"
                if self.resume_mapping:
                    self.resume_mapping = False
                    self.mapping = True
                    self.status = f"restored the unsaved map ({score:.0%} fit); still mapping"
            else:
                self.status = (f"could not line up confidently ({score:.0%} fit, next best {other:.0%}); "
                               "drive a little so the lidar sees more. Retrying every 20 s.")
        finally:
            self._aligning = False

    def info(self):
        return {"mode": self.mode(), "status": self.status, "align": self.align_info, "saved_at": self.saved_at,
                "aligning": self._aligning, "track": self.track_info}

    # ---------- for tests ----------
    def set_T(self, T):
        self.T = T
