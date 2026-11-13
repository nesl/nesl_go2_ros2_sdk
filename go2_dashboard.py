#!/usr/bin/env python3
"""Live Go2 sensor dashboard. Drive the dog from the page with the keyboard.

Drive controls accept requests from this machine and the lab subnet (CONTROL_NETS).
One person drives at a time; anyone can press Stop.
Recordings (ROS 2 bags) go to /recordings, mounted from the host recordings folder.

Run inside the driver container:  python3 /root/go2_dashboard.py [port]
Then open http://localhost:8080 . Tick Enable driving, click the page, and hold keys:
W/S forward/back, A/D strafe, Q/E turn, space stops.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")        # numpy's thread pools otherwise grab every core
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import ipaddress
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from go2_interfaces.msg import IMU, Go2State, WebRtcReq
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState, PointCloud2
from sensor_msgs_py import point_cloud2

from go2_mapper import Mapper
from go2_trail import Trail

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
CELL = 0.1            # metres per grid cell in the top view
OBSTACLE_MIN_H = 0.1  # cell counts as obstacle if its top is this far above the floor
MAP_MAX_TURN = 0.15   # rad/s; clouds taken while turning faster than this don't go on the map
CLOUD_RADIUS = 8.0   # metres around the robot shown in the top view

SPORT_TOPIC = "rt/api/sport/request"
AVOID_TOPIC = "rt/api/obstacles_avoid/request"
# obstacles_avoid service API ids (unitree_sdk2 ObstaclesAvoidClient)
AVOID_SWITCH_SET, AVOID_MOVE, AVOID_USE_API = 1001, 1003, 1004
# Drive commands are accepted from localhost and these subnets. The lab Wi-Fi is
# 172.17.0.0/18 (the dog is 172.17.39.82). Another site sets GO2_CONTROL_NETS.
CONTROL_NETS = [ipaddress.ip_network(n.strip()) for n in
                os.environ.get("GO2_CONTROL_NETS", "127.0.0.0/8,172.17.0.0/18").split(",")
                if n.strip()]
DRIVER_HOLD_S = 3.0        # a driver keeps control this long after their last command
CMD = {"StopMove": 1003, "StandUp": 1004, "StandDown": 1005, "BalanceStand": 1002, "Euler": 1007}
HEAD_MAX = math.radians(30)   # how far the body may tilt to point the camera up or down
UP_SIGN = -1                  # Euler pitch sign that raises the nose: untested on the dog, flip if Look up looks down
ACTIONS = {"stop": [], "stand": ["StandUp", "BalanceStand"], "lie": ["StandDown"]}
LIMITS = (0.3, 0.2, 0.6)   # max vx, vy (m/s), yaw rate (rad/s): same as go2_keys.py
DEADMAN_S = 0.3            # stop if the browser stops sending moves for this long
AVOID_DEFAULT = True       # the dog's obstacle avoidance starts on
NODE = None
MAPPER = None              # lab map built from the lidar clouds (go2_mapper.py)
TRAIL = None               # camera frames along the way, with map poses (go2_trail.py)
MAPS_DIR = "/maps"

REC_DIR = "/recordings"
REC_TOPICS = {"camera": ["/camera/image_raw", "/camera/camera_info"],
              "lidar": ["/point_cloud2"],
              "pose": ["/odom", "/imu", "/tf", "/tf_static", "/joint_states", "/go2_states"]}
REC_MIN_FREE_GB = 20       # refuse to start below this, stop automatically below half of it
HOST_UID = 1000            # recordings are handed to the host user when they finish


class Recorder:
    """Runs `ros2 bag record` for the chosen sensor groups."""

    def __init__(self):
        self.proc, self.name, self.groups, self.started = None, None, [], 0.0
        self.size, self.size_t, self.by, self.error = 0, 0.0, None, None
        self.lock = threading.Lock()

    def free_gb(self):
        return shutil.disk_usage(REC_DIR).free / 1e9

    def start(self, groups, by):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return "Already recording"
            groups = [g for g in REC_TOPICS if g in groups]
            if not any(g in groups for g in ("camera", "lidar")):
                return "Choose camera, lidar or both in Settings"
            if self.free_gb() < REC_MIN_FREE_GB:
                return f"Less than {REC_MIN_FREE_GB} GB free on disk"
            self.name = time.strftime("go2_%Y%m%d_%H%M%S_") + "_".join(g for g in groups if g != "pose")
            topics = " ".join(t for g in groups for t in REC_TOPICS[g])
            env = dict(os.environ, HOME="/tmp", ROS_LOG_DIR="/tmp/roslog_rec")
            # Runs as root like the driver: ROS's shared-memory transport delivers nothing between
            # different users, so a recorder running as the host user saved empty bags.
            self.proc = subprocess.Popen(
                ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash"
                 f" && exec ros2 bag record -o {REC_DIR}/{self.name} {topics}"],
                env=env, stdout=subprocess.DEVNULL, stderr=open("/tmp/rec.log", "w"))
            self.handed = False
            self.groups, self.started, self.size, self.by, self.error = groups, time.time(), 0, by, None
            return None

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.send_signal(signal.SIGINT)   # lets rosbag2 finish writing metadata
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.hand_over()

    def hand_over(self):
        """Give a finished recording to the host user."""
        if not self.name or getattr(self, "handed", True) or (self.proc and self.proc.poll() is None):
            return
        self.handed = True
        for d, dirs, files in os.walk(os.path.join(REC_DIR, self.name)):
            for f in [d] + [os.path.join(d, x) for x in files]:
                try:
                    os.chown(f, HOST_UID, HOST_UID)
                except OSError:
                    pass

    def dir_size(self, path):
        return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)

    def status(self):
        running = bool(self.proc and self.proc.poll() is None)
        if not running:
            self.hand_over()      # it may have ended by itself
        if self.proc and not running and self.started and self.error is None and time.time() - self.started < 5:
            self.error = "Recorder exited right away; see /tmp/rec.log in the container"
        now = time.time()
        if now - self.size_t > 1.0:
            self.size_t = now
            if running:
                self.size = self.dir_size(os.path.join(REC_DIR, self.name))
                if self.free_gb() < REC_MIN_FREE_GB / 2:
                    self.error = "Stopped: disk almost full"
                    threading.Thread(target=self.stop, daemon=True).start()
            self._past = sorted(
                ({"name": n, "mb": round(self.dir_size(os.path.join(REC_DIR, n)) / 1e6, 1)}
                 for n in os.listdir(REC_DIR) if os.path.isdir(os.path.join(REC_DIR, n))),
                key=lambda x: x["name"], reverse=True)[:8]
            self._free = round(self.free_gb())
        return {"running": running, "name": self.name, "groups": self.groups, "by": self.by,
                "secs": round(now - self.started) if running else 0, "mb": round(self.size / 1e6, 1),
                "error": self.error, "past": getattr(self, "_past", []), "free_gb": getattr(self, "_free", None)}


REC = Recorder()

lock = threading.Lock()
latest = {"imu": None, "odom": None, "state": None, "joints": None, "cloud": None,
          "map": None, "map_pose": None, "stamps": {}}
map_png = {"data": None}         # latest lab map as PNG, for /map.png
jpeg = {"data": None, "t": 0.0}
full = {"img": None, "t": 0.0}   # latest camera frame at full size, for /frame.jpg
jpeg_cond = threading.Condition()


class Places:
    """Named spots on the lab map, stored next to the map file. The page marks them; nothing drives there."""

    def __init__(self):
        self.lock = threading.Lock()

    def path(self):
        return os.path.join(MAPS_DIR, f"{os.environ.get('MAP_NAME', 'lab')}.places.json")

    def all(self):
        try:
            with open(self.path()) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def save(self, pid, description, by, pose=None, image=None, target=None):
        if not re.fullmatch(r"[a-z0-9_]{1,40}", pid or ""):
            raise ValueError("id must be 1-40 characters of a-z, 0-9 or _")
        if pose is None:
            with lock:
                mp = latest["map_pose"]
                img = full["img"]
            if mp is None:
                raise ValueError("no map pose: mapping or localization is not running")
            mp = dict(mp, pitch_deg=round(math.degrees(NODE.head if NODE else 0.0)))
        else:
            mp, img = pose, image
        name = os.environ.get("MAP_NAME", "lab")
        prev_dir = os.path.join(MAPS_DIR, f"{name}_places")
        os.makedirs(prev_dir, exist_ok=True)
        preview = None
        stale = os.path.join(prev_dir, f"{pid}.jpg")
        if img is None and os.path.exists(stale):
            os.remove(stale)
        if img is not None:
            preview = f"{pid}.jpg"
            cv2.imwrite(os.path.join(prev_dir, preview), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            try:
                os.chown(os.path.join(prev_dir, preview), HOST_UID, HOST_UID)
            except OSError:
                pass
        with self.lock:
            places = self.all()
            places[pid] = {"id": pid, "x": round(mp["x"], 3), "y": round(mp["y"], 3),
                           "yaw_deg": round(math.degrees(mp["yaw"]), 1),
                           "pitch_deg": float(mp.get("pitch_deg", 0.0)), "description": str(description or "")[:300],
                           "target": {"x": round(target[0], 2), "y": round(target[1], 2)} if target else None,
                           "preview": f"/places/{preview}" if preview else None, "map": name,
                           "saved_by": by, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "marked_on_map": pose is not None}
            self._write(places)
        try:
            os.chown(prev_dir, HOST_UID, HOST_UID)
        except OSError:
            pass
        return places[pid]

    def delete(self, pid):
        with self.lock:
            places = self.all()
            if places.pop(pid, None) is None:
                return False
            self._write(places)
            return True

    def _write(self, places):
        tmp = self.path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(places, f, indent=1)
        try:
            os.chown(tmp, HOST_UID, HOST_UID)
        except OSError:
            pass
        os.replace(tmp, self.path())


PLACES = Places()


def r(x, n=3):
    return round(float(x), n)


class Collector(Node):
    def __init__(self):
        super().__init__("go2_dashboard")
        q = qos_profile_sensor_data
        self.create_subscription(IMU, "/imu", self.on_imu, q)
        self.create_subscription(Odometry, "/odom", self.on_odom, q)
        self.create_subscription(Go2State, "/go2_states", self.on_state, q)
        self.create_subscription(JointState, "/joint_states", self.on_joints, q)
        self.create_subscription(PointCloud2, "/point_cloud2", self.on_cloud, q)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, q)
        self.create_timer(0.2, self.update_map_pose)
        self.create_timer(2.0, self.refresh_map)
        self.create_timer(20.0, lambda: MAPPER.autosave(latest["odom"]))
        self.create_timer(5.0, lambda: MAPPER.save_session(latest["odom"]))
        self._map_key = None
        self.last_cloud = 0.0
        self.vel_pub = self.create_publisher(Twist, "cmd_vel_out", 10)
        self.req_pub = self.create_publisher(WebRtcReq, "webrtc_req", 10)
        self.moving, self.last_move = False, 0.0
        self.avoid = False
        self.head = 0.0          # camera tilt from body pitch, rad, + = looking up
        self.driver, self.driver_ip, self.driver_t = None, None, 0.0
        self.create_timer(0.05, self.deadman)
        self._sync = self.create_timer(2.0, self.sync_avoid)

    def sync_avoid(self):
        """Once at startup: turn the dog's obstacle avoidance on (the console's default), so its
        state matches ours. Untick it on the console to drive without it."""
        self._sync.cancel()
        self.set_avoid(AVOID_DEFAULT)

    def req(self, api_id, topic, param="", priority=0):
        msg = WebRtcReq()
        msg.api_id, msg.topic, msg.priority = api_id, topic, priority
        msg.parameter = param if isinstance(param, str) else json.dumps(param)
        self.req_pub.publish(msg)

    def send(self, name):
        self.req(CMD[name], SPORT_TOPIC, "", 1 if name == "StopMove" else 0)

    def stop(self):
        if self.avoid:
            self.req(AVOID_MOVE, AVOID_TOPIC, {"x": 0.0, "y": 0.0, "yaw": 0.0, "mode": 0}, 1)
        self.send("StopMove")
        if self.head:        # StopMove may reset the body pose; keep looking where we were
            self.send_head()

    def send_head(self):
        self.req(CMD["Euler"], SPORT_TOPIC, {"x": 0.0, "y": round(UP_SIGN * self.head, 3), "z": 0.0})

    def look(self, rad):
        """Tilt the body so the camera looks up (+) or down (-); only while standing still."""
        self.head = max(-HEAD_MAX, min(HEAD_MAX, float(rad)))
        self.moving = False
        self.send("BalanceStand")
        time.sleep(0.1)
        self.send_head()

    def claim(self, client, ip):
        """One driver at a time: True if this client may drive now."""
        now = time.monotonic()
        if self.driver not in (None, client) and now - self.driver_t < DRIVER_HOLD_S:
            return False
        self.driver, self.driver_ip, self.driver_t = client, ip, now
        return True

    def move(self, x, y, z):
        x, y, z = (max(-l, min(l, float(v))) for v, l in zip((x, y, z), LIMITS))
        if self.head and (x or y or z):     # level off before walking
            self.head = 0.0
            self.send_head()
        if self.avoid:
            # Routed through the dog's obstacle-avoidance service, which slows/stops near obstacles
            self.req(AVOID_MOVE, AVOID_TOPIC, {"x": round(x, 2), "y": round(y, 2), "yaw": round(z, 2), "mode": 0})
        else:
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = x, y, z
            self.vel_pub.publish(t)
        self.moving, self.last_move = True, time.monotonic()

    def set_avoid(self, on):
        self.moving = False
        self.stop()
        self.avoid = bool(on)
        self.req(AVOID_SWITCH_SET, AVOID_TOPIC, {"enable": self.avoid})
        time.sleep(0.2)
        self.req(AVOID_USE_API, AVOID_TOPIC, {"is_remote_commands_from_api": self.avoid})

    def action(self, name):
        self.moving = False
        self.head = 0.0      # standing up or lying down resets the body pose
        self.stop()
        for c in ACTIONS[name]:
            time.sleep(0.2)
            self.send(c)

    def deadman(self):
        if self.moving and time.monotonic() - self.last_move > DEADMAN_S:
            self.moving = False
            self.stop()

    def stamp(self, key):
        latest["stamps"][key] = time.time()

    def on_imu(self, m):
        with lock:
            latest["imu"] = {"rpy": [r(v) for v in m.rpy],
                             "gyro": [r(v) for v in m.gyroscope],
                             "acc": [r(v) for v in m.accelerometer],
                             "temp": int(m.temperature)}
            self.stamp("imu")

    def on_odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        vel = self.odom_vel(time.monotonic(), p.x, p.y, yaw)
        with lock:
            latest["odom"] = {"pos": [r(p.x), r(p.y), r(p.z)], "yaw": r(yaw),
                              "vel": [r(v) for v in vel], "frame": m.header.frame_id}
            self.stamp("odom")

    def odom_vel(self, t, x, y, yaw):
        """Body-frame (vx, vy, wz) from successive odometry poses. The Go2 driver leaves the twist in
        /odom empty, so without this everything that checks whether the dog is turning saw zero."""
        prev = getattr(self, "_odom_prev", None)
        vel = getattr(self, "_odom_vel", [0.0, 0.0, 0.0])
        if prev is None or t - prev[0] > 1.0:
            self._odom_prev, self._odom_vel = (t, x, y, yaw), [0.0, 0.0, 0.0]
            return self._odom_vel
        dt = t - prev[0]
        if dt < 0.04:           # wait for a longer step: tiny ones are mostly noise
            return vel
        dx, dy = x - prev[1], y - prev[2]
        c, s = math.cos(prev[3]), math.sin(prev[3])
        raw = [(c * dx + s * dy) / dt, (-s * dx + c * dy) / dt,
               ((yaw - prev[3] + math.pi) % (2 * math.pi) - math.pi) / dt]
        a = min(1.0, dt / 0.15)  # smooth over ~0.15 s
        self._odom_vel = [v + a * (w - v) for v, w in zip(vel, raw)]
        self._odom_prev = (t, x, y, yaw)
        return self._odom_vel

    def on_state(self, m):
        with lock:
            latest["state"] = {"mode": int(m.mode), "gait": int(m.gait_type),
                               "body_height": r(m.body_height),
                               "foot_raise": r(m.foot_raise_height),
                               "velocity": [r(v) for v in m.velocity],
                               "obstacle": [r(v) for v in m.range_obstacle],
                               "foot_force": [int(v) for v in m.foot_force]}
            self.stamp("state")

    def on_joints(self, m):
        with lock:
            latest["joints"] = {"name": list(m.name),
                                "pos": [r(v) for v in m.position]}
            self.stamp("joints")

    def on_cloud(self, m):
        now = time.time()
        if now - self.last_cloud < 0.5:   # 2 Hz is plenty for the top view
            return
        self.last_cloud = now
        pts = point_cloud2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True)
        if pts.size == 0:
            return
        with lock:
            odom = latest["odom"]
        cx, cy = (odom["pos"][0], odom["pos"][1]) if odom and m.header.frame_id != "base_link" else (0.0, 0.0)
        near = pts[(np.abs(pts[:, 0] - cx) < CLOUD_RADIUS) & (np.abs(pts[:, 1] - cy) < CLOUD_RADIUS)]
        if len(near) == 0:
            return
        # Collapse to a fixed 2D grid (no random sampling, so the view is stable frame to frame).
        # Each cell keeps its highest point; cells well above the floor count as obstacles.
        floor = float(np.percentile(near[:, 2], 20))
        # The dog's own cloud smears walls into arcs while it turns, so only map while it isn't turning.
        if MAPPER and m.header.frame_id != "base_link" and odom and abs(odom["vel"][2]) < MAP_MAX_TURN:
            MAPPER.add_cloud(near, floor, (odom["pos"][0], odom["pos"][1]))
        ij = np.floor(near[:, :2] / CELL).astype(np.int64)
        key = ij[:, 0] * 100003 + ij[:, 1]
        order = np.lexsort((near[:, 2], key))
        last = np.r_[key[order][1:] != key[order][:-1], True]
        top = order[last]
        cells = np.c_[(ij[top] + 0.5) * CELL, near[top, 2] - floor]
        obs = cells[(cells[:, 2] > OBSTACLE_MIN_H) & (cells[:, 2] < 2.0)]
        flat = cells[cells[:, 2] <= OBSTACLE_MIN_H]
        with lock:
            latest["cloud"] = {"frame": m.header.frame_id, "n_total": int(len(pts)), "cell": CELL,
                               "floor_z": r(floor, 2),
                               "obs": np.round(obs.astype(np.float64), 2).tolist(),
                               "flat": np.round(flat[:, :2].astype(np.float64), 2).tolist()}
            self.stamp("cloud")

    def refresh_map(self):
        """Re-render the lab map when it or its alignment changed."""
        key = (MAPPER.version, MAPPER.T, MAPPER.status, (MAPPER.track_info or {}).get("at"))
        if key == self._map_key:
            return
        self._map_key = key
        png, meta = MAPPER.png()
        with lock:
            map_png["data"] = png
            latest["map"] = meta
            if meta:
                self.stamp("map")

    def update_map_pose(self):
        """Robot pose on the lab map; also lines a new session up with the saved map when needed."""
        with lock:
            odom = latest["odom"]
        MAPPER.try_align(odom)
        pose = MAPPER.pose(odom)
        with lock:
            latest["map_pose"] = pose
            if pose:
                self.stamp("map_pose")
        with jpeg_cond:
            data, t = jpeg["data"], jpeg["t"]
        TRAIL.offer(pose, data, t, odom["vel"][2] if odom else 0.0)

    def on_image(self, m):
        if m.encoding not in ("bgr8", "rgb8"):
            return
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        if m.encoding == "rgb8":
            img = img[:, :, ::-1]
        with lock:
            full["img"], full["t"] = img, time.time()
        img = cv2.resize(img, (640, int(640 * m.height / m.width)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with jpeg_cond:
                jpeg["data"], jpeg["t"] = buf.tobytes(), time.time()
                jpeg_cond.notify_all()
            with lock:
                self.stamp("camera")


def snapshot():
    """Everything the console shows, as one JSON-able dict (sent on /events and /state.json)."""
    with lock:
        now = time.time()
        snap = {k: v for k, v in latest.items() if k != "stamps"}
        snap["age"] = {k: r(now - t, 2) for k, t in latest["stamps"].items()}
    mono = time.monotonic()
    snap["rec"] = REC.status()
    snap["control"] = {"avoid": NODE.avoid, "look_deg": round(math.degrees(NODE.head)),
                       "driver": NODE.driver if mono - NODE.driver_t < DRIVER_HOLD_S else None,
                       "driver_ip": NODE.driver_ip}
    return snap


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def can_control(self):
        try:
            ip = ipaddress.ip_address(self.client_address[0].removeprefix("::ffff:"))
        except ValueError:
            return False
        return ip.is_loopback or any(ip in n for n in CONTROL_NETS)

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body, ctype):
        if body is None:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = self.path.split("?")[0]
        if not (path in ("/cmd", "/map") or path.startswith("/places")):
            return self.send_error(404)
        if not self.can_control():
            return self.send_error(403, "Drive controls only work from the lab network")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError as e:
            return self.send_error(400, str(e))
        if path == "/places":
            # {id, description} saves where the dog is now. Marking on the map instead:
            # target {x, y} = where the thing is; stand {x, y, yaw_deg} = where the dog should stand to
            # see it (e.g. where it stood for the chosen camera view), else picked automatically;
            # view = that camera view's URL, for the picture.
            try:
                pose = image = target = None
                t, st = req.get("target"), req.get("stand")
                if st is None and req.get("x") is not None:          # older form: x, y, yaw_deg
                    st = {"x": req["x"], "y": req["y"], "yaw_deg": req.get("yaw_deg")}
                if t is not None:
                    target = (float(t["x"]), float(t["y"]))
                if st is not None:
                    pose = {"x": float(st["x"]), "y": float(st["y"]), "yaw": math.radians(float(st.get("yaw_deg") or 0.0))}
                elif target is not None:
                    pose = {"x": target[0], "y": target[1], "yaw": 0.0}
                if pose is not None and req.get("view"):
                    f = TRAIL.path(str(req["view"]).removeprefix("/trail/"))
                    if f is None:
                        raise ValueError("that camera view no longer exists")
                    image = cv2.imread(f)
                return self.send_json(200, PLACES.save(str(req.get("id", "")), req.get("description", ""),
                                                       str(req.get("client") or self.client_address[0]),
                                                       pose=pose, image=image, target=target))
            except (ValueError, TypeError, KeyError) as e:
                return self.send_json(400, {"result": "error", "reason": str(e)})
        if path == "/map":
            act = req.get("action")
            with lock:
                odom = latest["odom"]
            if act == "start":     # extend the saved map when lined up with it, else a new one
                err = MAPPER.start_mapping(odom, fresh=MAPPER.T is None or not MAPPER.seen.any())
            elif act == "clear":
                err = MAPPER.clear()
            elif act == "frame":    # set the odom -> map transform by hand: {"T": [tx, ty, theta_rad]}
                MAPPER.T, err = tuple(float(v) for v in req["T"][:3]), None
                MAPPER.pending, MAPPER.resume_mapping = None, False
            elif act in ("new", "extend"):
                err = MAPPER.start_mapping(odom, fresh=act == "new")
            elif act == "stop":
                err = MAPPER.stop_mapping()
            elif act == "revert":
                err = MAPPER.revert()
            elif act == "save":
                err = MAPPER.save(odom)
            elif act == "align":
                err = None
                MAPPER.T = None
                MAPPER.try_align(odom, force=True)
            else:
                return self.send_json(400, {"result": "error", "reason": "action must be start, new, extend, stop, save, revert, clear, frame or align"})
            NODE._map_key = None
            if err:
                return self.send_json(409, {"result": "error", "reason": err})
            return self.send_json(200, {"result": "done", **MAPPER.info()})
        if path == "/places/delete":
            return self.send_json(200, {"result": "done" if PLACES.delete(str(req.get("id", ""))) else "not_found"})
        try:
            client, ip = str(req.get("client", "")), self.client_address[0]
            if "record" in req:                      # recording is independent of driving
                if req["record"]:
                    err = REC.start(req.get("groups", []), ip)
                    if err:
                        return self.send_error(409, err)
                else:
                    REC.stop()
            elif req.get("action") == "stop":        # anyone may stop the dog, always
                NODE.moving = False
                NODE.stop()
            elif not NODE.claim(client, ip):
                return self.send_error(409, f"Someone at {NODE.driver_ip} is driving")
            elif "move" in req:
                NODE.move(*req["move"][:3])
            elif "avoid" in req:
                NODE.set_avoid(req["avoid"])
            elif "look_deg" in req:          # camera tilt, + = up
                NODE.look(math.radians(float(req["look_deg"])))
            elif req.get("action") in ACTIONS:
                NODE.action(req["action"])
            else:
                return self.send_error(400, "Expected move, action, avoid or look_deg")
        except (ValueError, TypeError) as e:
            return self.send_error(400, str(e))
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path == "/control":
            body = json.dumps({"allowed": self.can_control(), "you": self.client_address[0]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"data: " + json.dumps(snapshot()).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path.split("?")[0] == "/state.json":
            self.send_json(200, snapshot())
        elif self.path.split("?")[0] == "/places":
            self.send_json(200, PLACES.all())
        elif self.path.startswith("/places/") and self.path.endswith(".jpg"):
            name = os.path.basename(self.path)
            try:
                with open(os.path.join("/maps", f"{os.environ.get('MAP_NAME', 'lab')}_places", name), "rb") as fh:
                    self.send_bytes(fh.read(), "image/jpeg")
            except OSError:
                self.send_error(404)
        elif self.path.split("?")[0] == "/trail/near":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                x, y = float(q["x"][0]), float(q["y"][0])
            except (KeyError, ValueError):
                return self.send_error(400, "x and y are required")
            self.send_json(200, {"views": TRAIL.near(x, y), "spot": MAPPER.cell_state(x, y)})
        elif self.path.split("?")[0] == "/trail/coverage":
            self.send_json(200, TRAIL.coverage())
        elif self.path.startswith("/trail/") and self.path.endswith(".jpg"):
            f = TRAIL.path(self.path[len("/trail/"):])
            if f is None:
                return self.send_error(404)
            with open(f, "rb") as fh:
                self.send_bytes(fh.read(), "image/jpeg")
        elif self.path.split("?")[0] == "/map.png":
            with lock:
                body = map_png["data"]
            if body is None:
                return self.send_error(503, "no map: mapping or localization is not running")
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.split("?")[0] == "/frame.jpg":
            # One full-size frame (1280x720 on the Go2), for vision models that need detail.
            with lock:
                img, t = full["img"], full["t"]
            if img is None:
                self.send_error(503, "no camera frame yet")
                return
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            body = buf.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Age", f"{time.time() - t:.2f}")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.split("?")[0] == "/camera.jpg":
            # Latest stream frame, newer than ?after=<t>; waits up to 5 s, else 204. The page
            # loops on this, so it picks the camera back up by itself after any restart.
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                after = float(q.get("after", ["0"])[0])
            except ValueError:
                after = 0.0
            with jpeg_cond:
                jpeg_cond.wait_for(lambda: jpeg["data"] is not None and jpeg["t"] > after, timeout=5)
                data, t = jpeg["data"], jpeg["t"]
            if data is None or t <= after:
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-T", repr(t))
            self.end_headers()
            self.wfile.write(data)
        elif self.path.split("?")[0] == "/camera.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = 0.0
            try:
                while True:
                    with jpeg_cond:
                        jpeg_cond.wait_for(lambda: jpeg["t"] > last, timeout=5)
                        data, last = jpeg["data"], jpeg["t"]
                    if data is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Go2 Live Sensors</title>
<style>
:root{--bg:#0f1214;--panel:#171b1e;--line:#262c31;--fg:#dfe5e8;--dim:#7d8a92;--accent:#4fb3a9;
--warn:#e0a84a;--bad:#e06a5a;--x:#e06a5a;--y:#7cc46d;--z:#5b9be0;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 var(--sans)}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;padding:14px 16px;border-bottom:1px solid var(--line)}
h1{font-size:17px;margin:0;font-weight:600;letter-spacing:.01em}
.sub{color:var(--dim);font-size:12px}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-left:auto}
.chip{font:11px var(--mono);padding:3px 8px;border-radius:99px;border:1px solid var(--line);color:var(--dim)}
.chip.ok{color:var(--accent);border-color:color-mix(in srgb,var(--accent) 45%,transparent)}
.chip.stale{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,transparent)}
.chip.dead{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 45%,transparent)}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;padding:12px 16px 24px}
section{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px;min-width:0}
h2{margin:0 0 10px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);font-weight:600;display:flex;justify-content:space-between}
h2 span{font:11px var(--mono);letter-spacing:0;text-transform:none}
.wide{grid-column:span 2}@media (max-width:700px){.wide{grid-column:auto}}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font:13px var(--mono);font-variant-numeric:tabular-nums}
.kv dt{color:var(--dim)}.kv dd{margin:0;text-align:right}
canvas{width:100%;display:block}
#cam{width:100%;max-height:78vh;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:3px}
.feet{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.foot{font:12px var(--mono)}.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;margin-top:3px}
.bar i{display:block;height:100%;background:var(--accent);width:0;transition:width .1s}
table{width:100%;border-collapse:collapse;font:12px var(--mono);font-variant-numeric:tabular-nums}
th{color:var(--dim);font-weight:400;text-align:right;padding:2px 4px}th:first-child,td:first-child{text-align:left}
td{text-align:right;padding:2px 4px;border-top:1px solid var(--line)}
.legend{font:11px var(--mono);color:var(--dim);display:flex;gap:10px;margin-top:6px}
.drive{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}
.pad{display:grid;grid-template-columns:repeat(3,64px);grid-template-rows:repeat(3,52px);gap:6px;touch-action:none;user-select:none}
.pad button,.acts button{font:600 13px var(--sans);color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;cursor:pointer}
.pad button small{display:block;font:11px var(--mono);color:var(--dim);font-weight:400}
.pad button.on{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.pad button.on small{color:var(--bg)}
#bstop{background:color-mix(in srgb,var(--bad) 25%,var(--bg));border-color:var(--bad)}
.acts{display:grid;grid-template-columns:repeat(3,auto);gap:6px}
.looks{display:grid;gap:6px}.looks small{color:var(--dim);font-weight:400}#lookstate{font:12px var(--mono);color:var(--dim)}.acts button{padding:9px 12px}
button:focus-visible,input:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button:disabled{opacity:.4;cursor:not-allowed}
.arm{display:flex;gap:8px;align-items:center;font-weight:600}.arm input{width:18px;height:18px}
.dstat{font:12px var(--mono);color:var(--dim);min-height:1.4em}.dstat.live{color:var(--accent)}
.help{font-size:12px;color:var(--dim);max-width:30ch;display:grid;gap:6px}
.recrow{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
#recbtn{display:inline-flex;gap:8px;align-items:center;font:600 14px var(--sans);padding:10px 16px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
#recbtn i{width:12px;height:12px;border-radius:50%;background:var(--bad)}
#recbtn.on{background:var(--bad);border-color:var(--bad);color:#fff}#recbtn.on i{background:#fff;border-radius:2px}
#gear{font:13px var(--sans);padding:9px 12px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
.recinfo{font:13px var(--mono);font-variant-numeric:tabular-nums;color:var(--dim)}.recinfo b{color:var(--fg);font-weight:500}
#settings{margin-top:12px;padding:12px;border:1px solid var(--line);border-radius:6px;display:grid;gap:8px;max-width:520px}
#settings label{display:flex;gap:8px;align-items:baseline}#settings small{color:var(--dim)}
.past{margin-top:10px;font:12px var(--mono);color:var(--dim);display:grid;gap:2px}
.err{color:var(--bad)}
.full{grid-column:1/-1}
.mapctl{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:8px}
.mapctl button{font:12px var(--sans);padding:5px 9px;border-radius:5px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
.mapstatus{flex-basis:100%;font:12px var(--mono);color:var(--dim)}
.maprow{display:grid;gap:6px;border-top:1px solid var(--line);padding-top:10px}
.mapstate{font:12px var(--mono);color:var(--dim)}.mapstate.on{color:var(--accent)}
.mapbtns{display:flex;flex-wrap:wrap;gap:6px}
.mapbtns button{font:600 13px var(--sans);color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;cursor:pointer;padding:9px 12px}
.pl{display:grid;grid-template-columns:64px 1fr auto;gap:8px;align-items:center;font-size:13px;padding:4px 0;border-top:1px solid var(--line)}
.pl img{width:64px;aspect-ratio:16/9;object-fit:cover;border-radius:3px;background:#000}.pl small{color:var(--dim);font:11px var(--mono);display:block}
.pl button,#plform button{font:12px var(--sans);padding:5px 9px;border-radius:5px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
#plform,#dropform{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
#map{cursor:crosshair}
#drop{margin-top:10px;padding-top:10px;border-top:1px solid var(--line);font-size:13px}
#drop .droph{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
#drop small{color:var(--dim);font:11px var(--mono)}
#dropbig{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:4px;margin-top:8px;display:block}
#dropviews{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:6px;margin-top:8px}
#dropviews figure{margin:0;cursor:pointer;border:2px solid transparent;border-radius:4px}
#dropviews figure.sel{border-color:var(--accent)}
#dropviews img{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;border-radius:2px;background:#000}
#dropviews figcaption{font:10px var(--mono);color:var(--dim);padding:2px}
#drop button,#dropform button{font:12px var(--sans);padding:5px 9px;border-radius:5px;border:1px solid var(--line);background:var(--bg);color:var(--fg);cursor:pointer}
#dropform input[type=text],#plform input{flex:1 1 120px;min-width:0;font:13px var(--sans);padding:6px 8px;border-radius:5px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
.legend b{font-weight:400}.lx{color:var(--x)}.ly{color:var(--y)}.lz{color:var(--z)}
</style></head><body>
<header><h1>Go2 Live Sensors</h1><span class="sub" id="conn">connecting…</span>
<div class="chips" id="chips"></div></header>
<main>
<section class="full"><h2>Live camera <span id="camage"></span></h2><img id="cam" alt="Front camera stream"></section>
<section class="wide"><h2>Drive <span id="dnote"></span></h2><div class="drive">
<div class="pad" id="pad">
<button data-k="q">↺<small>Q</small></button><button data-k="w">↑<small>W</small></button><button data-k="e">↻<small>E</small></button>
<button data-k="a">←<small>A</small></button><button id="bstop">STOP<small>space</small></button><button data-k="d">→<small>D</small></button>
<span></span><button data-k="s">↓<small>S</small></button><span></span></div>
<div style="display:grid;gap:10px">
<label class="arm"><input type="checkbox" id="armed"> Enable driving</label>
<label class="arm"><input type="checkbox" id="avoid"> Obstacle avoidance</label>
<div class="acts"><button data-a="stand">Stand up</button><button data-a="lie">Lie down</button></div>
<div class="looks"><strong>Camera tilt</strong> <span id="lookstate">level</span>
<div class="acts"><button data-look="6">Look up<small> R</small></button><button data-look="0">Level</button><button data-look="-6">Look down<small> F</small></button></div></div>
<div class="maprow"><strong>Mapping</strong> <span id="mapstate" class="mapstate">…</span>
<div class="mapbtns"><button data-map="start" data-when="idle">Start mapping</button><button data-map="save" data-when="mapping">Stop and save</button>
<button data-map="clear" data-when="any">Clear map</button><button data-map="revert" data-when="any">Back to saved map</button></div></div>
<div class="dstat" id="dstat">Disarmed</div></div>
<div class="help"><span>Tick Enable driving first; it's a safety switch so stray clicks and keys do nothing.</span>
<span>Hold a direction to move; letting go stops the dog within 0.3 s. Keys work while this tab is focused.</span>
<span>Max 0.3 m/s forward, 0.2 m/s sideways. Keep someone next to the dog.</span>
<span>One person drives at a time. Anyone can press STOP.</span>
<span>Look up / down tilts the whole body while standing still, about 6° per press, up to 30°. It levels off by itself when you drive.</span>
<span>Obstacle avoidance makes the dog slow down and stop near obstacles.</span></div>
</div></section>
<section class="wide"><h2>Recording <span id="recnote"></span></h2>
<div class="recrow"><button id="recbtn"><i></i><span>Record</span></button>
<button id="gear" aria-expanded="false" aria-controls="settings">Settings</button>
<span class="recinfo" id="recinfo"></span></div>
<div id="settings" hidden>
<strong>What to record</strong>
<label><input type="checkbox" id="r_camera" checked> Camera <small>front camera, raw 1280×720 at ~5 fps, about 50 GB/hour</small></label>
<label><input type="checkbox" id="r_lidar" checked> Lidar <small>point cloud at ~7 Hz, about 25 GB/hour</small></label>
<label><input type="checkbox" id="r_pose" checked> Robot pose <small>odometry, IMU, joints, transforms; tiny, needed to line lidar frames up</small></label>
<small>Saved as ROS 2 bags in ~/oliver/go2_recordings on the control PC. These settings are remembered in this browser.</small>
</div>
<div class="past" id="past"></div></section>
<section><h2>Lidar, top view <span id="cloudinfo"></span></h2><canvas id="cloud" width="600" height="600"></canvas>
<div class="legend">grid lines every 2 m · grey = open floor seen · yellow→red = obstacle 10 cm→1.2 m tall · arrow = robot heading</div></section>
<section><h2>Map <span id="mapinfo"></span></h2><canvas id="map" width="600" height="600"></canvas>
<div class="legend" id="maplegend">white = open floor · black = walls and furniture · grey = not seen yet · arrow = robot · dots = places (line to where the dog stands to see it) · click the map to mark a place</div>
<div class="mapctl"><span id="mapstatus" class="mapstatus"></span>
<button data-map="new" data-when="idle">Start new map</button><button data-map="extend" data-when="idle">Extend map</button>
<button data-map="save" data-when="mapping">Stop and save</button><button data-map="stop" data-when="mapping">Stop without saving</button><button data-map="align" data-when="idle">Line up again</button><button data-map="revert" data-when="any">Back to saved map</button></div>
<div id="drop" hidden>
<div class="droph"><strong>New place at <span id="dropxy"></span></strong><span id="dropspot"></span><button id="dropx" type="button">Cancel</button></div>
<small id="dropvnote"></small>
<img id="dropbig" alt="Camera view picked for this place" hidden>
<div id="dropviews"></div>
<small id="dropstandnote"></small>
<form id="dropform"><input type="text" id="dropid" placeholder="name, e.g. shelf_2" pattern="[a-z0-9_]{1,40}" required><input type="text" id="droppdesc" placeholder="what is here"><button>Save place</button></form>
<small id="dropmsg"></small>
</div></section>
<section><h2>Places <span id="plnote"></span></h2><div id="places"></div>
<form id="plform"><input id="plid" type="text" placeholder="name, e.g. shelf_2" pattern="[a-z0-9_]{1,40}" required><input id="pldesc" type="text" placeholder="what is here"><button>Save this spot</button></form>
<div class="legend">Saves where the dog stands now. To mark a place later, click the map. Click a place here to show its camera cone.</div></section>
<section><h2>Orientation (IMU)</h2><canvas id="rpyplot" width="600" height="180"></canvas>
<div class="legend"><b class="lx">roll</b><b class="ly">pitch</b><b class="lz">yaw</b> · last 20 s, degrees</div>
<dl class="kv" id="imu" style="margin-top:10px"></dl></section>
<section><h2>Odometry</h2><dl class="kv" id="odom"></dl></section>
<section><h2>Body state</h2><dl class="kv" id="state"></dl></section>
<section><h2>Foot contact force</h2><div class="feet" id="feet"></div></section>
<section><h2>Joint angles (rad)</h2><table id="joints"></table></section>
</main>
<script>
const $=id=>document.getElementById(id);
const deg=v=>(v*180/Math.PI).toFixed(1);
const f=(v,n=2)=>(v>=0?' ':'')+Number(v).toFixed(n);
const kv=(el,rows)=>{el.innerHTML=rows.map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('')};
const hist=[];const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function plotRPY(){const c=$('rpyplot'),g=c.getContext('2d'),W=c.width,H=c.height;g.clearRect(0,0,W,H);
 const now=performance.now()/1000;g.strokeStyle=css('--line');g.lineWidth=1;
 [-30,0,30].forEach(d=>{const y=H/2-d/45*(H/2);g.beginPath();g.moveTo(0,y);g.lineTo(W,y);g.stroke();
  g.fillStyle=css('--dim');g.font='11px monospace';g.fillText(d+'°',4,y-3)});
 ['--x','--y','--z'].forEach((col,i)=>{g.strokeStyle=css(col);g.lineWidth=1.6;g.beginPath();
  hist.forEach((h,j)=>{const x=W-(now-h.t)/20*W,y=H/2-Math.max(-45,Math.min(45,h.v[i]*180/Math.PI))/45*(H/2);
   j?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()})}

function plotCloud(cl,od){const c=$('cloud'),g=c.getContext('2d'),W=c.width,R=8,s=W/(2*R);g.clearRect(0,0,W,W);
 g.strokeStyle=css('--line');g.lineWidth=1;for(let m=-R;m<=R;m+=2){const p=W/2+m*s;
  g.beginPath();g.moveTo(p,0);g.lineTo(p,W);g.moveTo(0,p);g.lineTo(W,p);g.stroke()}
 const local=cl&&cl.frame==='base_link';const cx=(!local&&od)?od.pos[0]:0,cy=(!local&&od)?od.pos[1]:0;
 if(cl){const c2=cl.cell*s;g.fillStyle=css('--line');
  for(const [x,y] of cl.flat)g.fillRect(W/2+(x-cx)*s-c2/2,W/2-(y-cy)*s-c2/2,c2,c2);
  for(const [x,y,h] of cl.obs){const t=Math.min(1,h/1.2);g.fillStyle=`hsl(${45-t*40} 90% ${62-t*10}%)`;
   g.fillRect(W/2+(x-cx)*s-c2/2,W/2-(y-cy)*s-c2/2,c2,c2)}}
 const yaw=local?0:(od?od.yaw:0);g.save();g.translate(W/2,W/2);g.rotate(-yaw);g.fillStyle=css('--fg');
 g.beginPath();g.moveTo(14,0);g.lineTo(-9,7);g.lineTo(-5,0);g.lineTo(-9,-7);g.closePath();g.fill();g.restore()}

const mapImg=new Image();let mapT=null;let PLACES={};let mapXf=null,DROP=null,COVER=[],SELPLACE=null;
function plotMap(m,p){const c=$('map'),g=c.getContext('2d'),W=c.width;g.clearRect(0,0,W,W);
 if(!m){$('mapinfo').textContent='no map';$('mapstatus').textContent='No map yet. Press Start mapping, then drive slowly around the lab.';
  $('mapstate').textContent='off · no map yet';document.querySelectorAll('[data-map]').forEach(b=>{const w=b.dataset.when;b.hidden=w==='mapping'||b.dataset.map==='extend'||b.dataset.map==='align'});return}
 $('mapstatus').textContent=m.status+(m.mode==='mapping'?' · drive slowly to fill it in':'');
 const mapping=m.mode==='mapping';
 $('mapstate').textContent=mapping?`on · ${m.mapped_m2} m² seen`:m.mode==='localized'?'off · lined up with the saved map':'off · '+m.status;$('mapstate').className='mapstate'+(mapping?' on':'');
 document.querySelectorAll('[data-map]').forEach(b=>{const w=b.dataset.when;b.hidden=w==='any'?false:(w==='mapping')!==mapping});
 if(m.t!==mapT){mapT=m.t;mapImg.src='/map.png?t='+m.t}
 if(!mapImg.complete||!mapImg.naturalWidth)return;
 const s=Math.min(W/m.width,W/m.height),ox=(W-m.width*s)/2,oy=(W-m.height*s)/2;
 g.imageSmoothingEnabled=false;g.drawImage(mapImg,ox,oy,m.width*s,m.height*s);
 $('mapinfo').textContent=`${(m.width*m.resolution).toFixed(0)}×${(m.height*m.resolution).toFixed(0)} m · ${m.mapped_m2} m² seen`;
 mapXf={s,ox,oy,m};const mp=(x,y)=>[ox+(x-m.origin[0])/m.resolution*s,oy+(m.height-(y-m.origin[1])/m.resolution)*s];
 const ppm=s/m.resolution;   // canvas px per metre
 const arrow=(x,y,yaw,col,sz=1)=>{g.save();g.translate(x,y);g.rotate(-yaw);g.scale(sz,sz);g.fillStyle=col;g.beginPath();g.moveTo(11,0);g.lineTo(-7,6);g.lineTo(-3,0);g.lineTo(-7,-6);g.closePath();g.fill();g.restore()};
 // camera view cone: what the front camera sees (about 40 degrees each side)
 const cone=(x,y,yaw,len,col)=>{g.save();g.fillStyle=col;g.globalAlpha=.18;g.beginPath();g.moveTo(x,y);g.arc(x,y,len,-yaw-0.7,-yaw+0.7);g.closePath();g.fill();
  g.globalAlpha=.6;g.strokeStyle=col;g.lineWidth=1;g.stroke();g.restore()};
 if(DROP){g.fillStyle=css('--z');g.globalAlpha=.35;for(const [x,y] of COVER){const [cx,cy]=mp(x,y);g.fillRect(cx-1.5,cy-1.5,3,3)}g.globalAlpha=1;
  const [dx,dy]=mp(DROP.x,DROP.y);
  const v=DROP.view;if(v){const [vx,vy]=mp(v.x,v.y);cone(vx,vy,v.yaw_deg*Math.PI/180,Math.max(Math.hypot(dx-vx,dy-vy)+0.5*ppm,1.5*ppm),css('--warn'));
   g.strokeStyle=css('--warn');g.setLineDash([4,3]);g.beginPath();g.moveTo(vx,vy);g.lineTo(dx,dy);g.stroke();g.setLineDash([]);arrow(vx,vy,v.yaw_deg*Math.PI/180,css('--warn'))}
  g.strokeStyle=css('--warn');g.lineWidth=2;g.beginPath();g.arc(dx,dy,8,0,7);g.moveTo(dx-12,dy);g.lineTo(dx+12,dy);g.moveTo(dx,dy-12);g.lineTo(dx,dy+12);g.stroke()}
 g.font='11px monospace';for(const q of Object.values(PLACES)){const [sx,sy]=mp(q.x,q.y),t=q.target;
  if(t){const [tx,ty]=mp(t.x,t.y);g.strokeStyle=css('--accent');g.globalAlpha=.5;g.lineWidth=1;g.beginPath();g.moveTo(sx,sy);g.lineTo(tx,ty);g.stroke();g.globalAlpha=1;
   if(q.id===SELPLACE)cone(sx,sy,q.yaw_deg*Math.PI/180,Math.hypot(tx-sx,ty-sy)+0.5*ppm,css('--accent'));
   arrow(sx,sy,q.yaw_deg*Math.PI/180,css('--accent'),.6);g.fillStyle=css('--accent');g.beginPath();g.arc(tx,ty,4.5,0,7);g.fill();g.fillText(q.id,tx+6,ty-4)}
  else{if(q.id===SELPLACE)cone(sx,sy,q.yaw_deg*Math.PI/180,1.5*ppm,css('--accent'));g.fillStyle=css('--accent');g.beginPath();g.arc(sx,sy,4,0,7);g.fill();g.fillText(q.id,sx+6,sy-4)}}
 if(p){const px=ox+(p.x-m.origin[0])/m.resolution*s,py=oy+(m.height-(p.y-m.origin[1])/m.resolution)*s;
  g.save();g.translate(px,py);g.rotate(-p.yaw);g.fillStyle=css('--bad');
  g.beginPath();g.moveTo(12,0);g.lineTo(-8,6);g.lineTo(-4,0);g.lineTo(-8,-6);g.closePath();g.fill();g.restore()}}
mapImg.onload=()=>plotMap(lastMap,lastPose);let lastMap=null,lastPose=null;
const FEET=['FR','FL','RR','RL'];const SENS=['imu','odom','state','joints','cloud','camera'];
function render(d){const a=d.age||{};
 if(d.rec)renderRec(d.rec);
 if(d.control){ctl=d.control;$('avoid').checked=ctl.avoid;
  const lk=ctl.look_deg||0,bp=d.imu?Math.round(d.imu.rpy[1]*180/Math.PI):null;
  $('lookstate').textContent=(lk?`${Math.abs(lk)}° ${lk>0?'up':'down'}`:'level')+(bp!=null?` · body pitch ${bp>0?'+':''}${bp}°`:'');
  const other=ctl.driver&&ctl.driver!==CLIENT;if(allowed)$('dnote').textContent=other?`driving: ${ctl.driver_ip}`:(ctl.driver?'you are driving':'');}
 $('chips').innerHTML=SENS.map(k=>{const t=a[k];const cls=t==null?'dead':t<1.5?'ok':t<5?'stale':'dead';
  return `<span class="chip ${cls}">${k} ${t==null?'—':t<1.5?'live':t.toFixed(0)+'s'}</span>`}).join('');
 if(d.imu){const i=d.imu;hist.push({t:performance.now()/1000,v:i.rpy});while(hist.length&&performance.now()/1000-hist[0].t>20)hist.shift();
  kv($('imu'),[['roll / pitch / yaw (°)',`${deg(i.rpy[0])} / ${deg(i.rpy[1])} / ${deg(i.rpy[2])}`],
   ['gyro (rad/s)',i.gyro.map(v=>f(v)).join(' ')],['accel (m/s²)',i.acc.map(v=>f(v)).join(' ')],['IMU temp (°C)',i.temp]])}
 plotRPY();
 if(d.odom){const o=d.odom;kv($('odom'),[['frame',o.frame],['x / y / z (m)',o.pos.map(v=>f(v)).join(' ')],
  ['heading (°)',deg(o.yaw)],['vx / vy (m/s)',f(o.vel[0])+' '+f(o.vel[1])],['turn rate (rad/s)',f(o.vel[2])]])}
 if(d.state){const s=d.state;kv($('state'),[['mode',s.mode],['gait type',s.gait],['body height (m)',f(s.body_height,3)],
  ['foot raise (m)',f(s.foot_raise,3)],['velocity (m/s)',s.velocity.map(v=>f(v)).join(' ')],
  ['obstacle range F/B/L/R (m)',s.obstacle.map(v=>f(v,1)).join(' ')]]);
  const mx=Math.max(100,...s.foot_force);$('feet').innerHTML=s.foot_force.map((v,k)=>
   `<div class="foot">${FEET[k]} <span style="float:right">${v}</span><div class="bar"><i style="width:${v/mx*100}%"></i></div></div>`).join('')}
 if(d.joints){const j=d.joints,legs={};j.name.forEach((n,k)=>{const [leg,part]=n.split('_');(legs[leg]??={})[part]=j.pos[k]});
  $('joints').innerHTML='<tr><th>leg</th><th>hip</th><th>thigh</th><th>calf</th></tr>'+Object.entries(legs).map(([l,p])=>
   `<tr><td>${l}</td><td>${f(p.hip)}</td><td>${f(p.thigh)}</td><td>${f(p.calf)}</td></tr>`).join('')}
 if(d.cloud)$('cloudinfo').textContent=`${d.cloud.obs.length} obstacle cells · ${d.cloud.n_total} pts`;
 plotCloud(d.cloud,d.odom);lastMap=d.map;lastPose=d.map_pose;plotMap(d.map,d.map_pose);$('camage').textContent=a.camera!=null?`${a.camera.toFixed(1)} s old`:'no frames'}

// The camera <img> stream does not reconnect by itself when the console restarts; restart it whenever the event stream (re)connects.
async function camLoop(){let after=0;const cam=$('cam');
 for(;;){try{const r=await fetch('/camera.jpg?after='+after,{cache:'no-store'});
   if(r.status===200){after=+r.headers.get('X-Frame-T')||0;const u=URL.createObjectURL(await r.blob()),old=cam.src;
     cam.src=u;if(old.startsWith('blob:'))URL.revokeObjectURL(old)}
   else if(r.status!==204)throw new Error(r.status)}
  catch(e){after=0;await new Promise(f=>setTimeout(f,1000))}}}
camLoop();
function connect(){const es=new EventSource('/events');es.onopen=()=>{$('conn').textContent='streaming from '+location.host};
 es.onmessage=e=>render(JSON.parse(e.data));es.onerror=()=>{$('conn').textContent='disconnected, retrying…'}}
connect();

const GROUPS=['camera','lidar','pose'];
try{const saved=JSON.parse(localStorage.getItem('recGroups')||'null');if(saved)GROUPS.forEach(g=>$('r_'+g).checked=saved.includes(g))}catch(e){}
GROUPS.forEach(g=>$('r_'+g).addEventListener('change',()=>{try{localStorage.setItem('recGroups',JSON.stringify(GROUPS.filter(x=>$('r_'+x).checked)))}catch(e){}}));
$('gear').addEventListener('click',()=>{const h=!$('settings').hidden;$('settings').hidden=h;$('gear').setAttribute('aria-expanded',String(!h))});
let recRunning=false;
$('recbtn').addEventListener('click',async()=>{const body=recRunning?{record:false}:{record:true,groups:GROUPS.filter(g=>$('r_'+g).checked)};
 $('recbtn').disabled=true;try{const r=await fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){const t=await r.text();const m=t.match(/<p>Message: (.*?)\./);$('recnote').innerHTML=`<span class="err">${m?m[1]:'Could not start ('+r.status+')'}</span>`}else $('recnote').textContent=''}
 finally{$('recbtn').disabled=false}});
const hms=s=>[Math.floor(s/3600),Math.floor(s/60)%60,s%60].map(v=>String(v).padStart(2,'0')).join(':');
function renderRec(r){recRunning=r.running;$('recbtn').classList.toggle('on',r.running);$('recbtn').querySelector('span').textContent=r.running?'Stop recording':'Record';
 $('recinfo').innerHTML=r.running?`<b>${hms(r.secs)}</b> · ${r.mb>1000?(r.mb/1000).toFixed(1)+' GB':r.mb+' MB'} · ${r.groups.join(', ')} · started from ${r.by}`
  :(r.error?`<span class="err">${r.error}</span> · `:'')+(r.free_gb!=null?`${r.free_gb} GB free`:'');
 $('past').innerHTML=r.past.length?'Saved recordings:<br>'+r.past.map(p=>`${p.name} · ${p.mb>1000?(p.mb/1000).toFixed(1)+' GB':p.mb+' MB'}`).join('<br>'):''}

const DIR={w:[1,0,0],s:[-1,0,0],a:[0,1,0],d:[0,-1,0],q:[0,0,1],e:[0,0,-1]},LIM=[0.3,0.2,0.6];
const held=new Set();let armed=false,allowed=false,wasMoving=false;
const CLIENT=Math.random().toString(36).slice(2);let ctl={};
const post=b=>fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...b,client:CLIENT,console:true})})
 .then(async r=>{if(r.status===409){held.clear();paint();$('dstat').textContent='Someone else is driving. Wait for them to stop.';return}
  if(!r.ok)throw new Error(r.status)}).catch(e=>{$('dstat').textContent='Command failed ('+e.message+'). Is the dashboard still running?'});
function setArmed(v){armed=v&&allowed;$('armed').checked=armed;held.clear();paint();
 document.querySelectorAll('#pad button,.acts button').forEach(b=>b.disabled=!armed&&b.id!=='bstop');$('avoid').disabled=!armed;
 $('dstat').textContent=armed?'Armed: hold a direction':'Disarmed';$('dstat').className='dstat'+(armed?' live':'')}
function paint(){document.querySelectorAll('#pad [data-k]').forEach(b=>b.classList.toggle('on',held.has(b.dataset.k)))}
function stop(){held.clear();paint();wasMoving=false;if(allowed)post({action:'stop'})}
setInterval(()=>{if(!armed)return;
 if(held.size){const v=[0,0,0];held.forEach(k=>DIR[k].forEach((d,i)=>v[i]+=d));const m=v.map((d,i)=>Math.sign(d)*LIM[i]);
  post({move:m});wasMoving=true;$('dstat').textContent=`moving  x=${m[0].toFixed(1)} y=${m[1].toFixed(1)} turn=${m[2].toFixed(1)}`}
 else if(wasMoving){wasMoving=false;$('dstat').textContent='Armed: stopped'}},100);
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.key!==' ')return;const k=e.key.toLowerCase();
 if(k===' '){e.preventDefault();stop();return}if(armed&&DIR[k]){e.preventDefault();held.add(k);paint()}
 if(!e.repeat&&(k==='r'||k==='f')){e.preventDefault();look(k==='r'?6:-6)}});
addEventListener('keyup',e=>{const k=e.key.toLowerCase();if(held.delete(k))paint()});
addEventListener('blur',()=>{if(held.size)stop()});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&held.size)stop()});
document.querySelectorAll('#pad [data-k]').forEach(b=>{const k=b.dataset.k;
 b.addEventListener('pointerdown',e=>{if(!armed)return;b.setPointerCapture(e.pointerId);held.add(k);paint()});
 ['pointerup','pointercancel','lostpointercapture'].forEach(ev=>b.addEventListener(ev,()=>{if(held.delete(k))paint()}))});
$('bstop').addEventListener('click',stop);
document.querySelectorAll('.acts [data-a]').forEach(b=>b.addEventListener('click',()=>{if(!armed)return;held.clear();paint();
 post({action:b.dataset.a});$('dstat').textContent=b.textContent}));
function look(step){if(!armed)return;held.clear();paint();const cur=ctl.look_deg||0,to=step?Math.max(-30,Math.min(30,cur+step)):0;
 ctl.look_deg=to;post({look_deg:to});$('dstat').textContent=to?`Camera tilted ${Math.abs(to)}° ${to>0?'up':'down'}`:'Camera level'}
document.querySelectorAll('[data-look]').forEach(b=>b.addEventListener('click',()=>look(+b.dataset.look)));
$('avoid').addEventListener('change',e=>{if(!armed){e.target.checked=!!ctl.avoid;return}held.clear();paint();post({avoid:e.target.checked});
 $('dstat').textContent='Obstacle avoidance '+(e.target.checked?'on':'off')});
$('armed').addEventListener('change',e=>{setArmed(e.target.checked);if(!e.target.checked)stop()});
fetch('/control').then(r=>r.json()).then(c=>{allowed=c.allowed;
 $('dnote').textContent=allowed?'':'view only: your computer is not on the lab network';$('armed').disabled=!allowed;setArmed(false)});
setArmed(false);

const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function loadPlaces(){try{PLACES=await (await fetch('/places')).json()}catch(e){return}
 const v=Object.values(PLACES);$('plnote').textContent=v.length?v.length+' saved':'';
 $('places').innerHTML=v.length?v.map(p=>`<div class="pl"><img src="${p.preview||''}" alt=""><div data-sel="${esc(p.id)}" style="cursor:pointer">${esc(p.id)}<small>${esc(p.description||'')}${p.target?` · thing at ${p.target.x.toFixed(1)}, ${p.target.y.toFixed(1)}`:''} · stand ${p.x.toFixed(1)}, ${p.y.toFixed(1)} facing ${Math.round(p.yaw_deg)}°</small></div><button data-del="${esc(p.id)}">Delete</button></div>`).join('')
  :'<div class="legend">No places yet.</div>';
 document.querySelectorAll('[data-sel]').forEach(d=>d.onclick=()=>{SELPLACE=SELPLACE===d.dataset.sel?null:d.dataset.sel;plotMap(lastMap,lastPose)});
 document.querySelectorAll('[data-del]').forEach(b=>b.onclick=async()=>{if(!confirm('Delete place '+b.dataset.del+'?'))return;
  await fetch('/places/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:b.dataset.del})});loadPlaces()})}
$('plform').addEventListener('submit',async e=>{e.preventDefault();
 const r=await fetch('/places',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:$('plid').value.trim(),description:$('pldesc').value,client:'console '+CLIENT})});
 if(!r.ok){const j=await r.json().catch(()=>({}));$('plnote').innerHTML=`<span class="err">${esc(j.reason||r.status)}</span>`;return}
 $('plid').value=$('pldesc').value='';loadPlaces()});
loadPlaces();setInterval(loadPlaces,5000);
const hhmm=t=>{const d=new Date(t*1000);return d.toLocaleDateString(undefined,{month:'short',day:'numeric'})+' '+d.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})};
const side=b=>Math.abs(b)<=10?'ahead':Math.abs(b)>=100?'behind':`${Math.abs(b)}° ${b>0?'left':'right'}`;
function pickView(v){DROP.view=v;$('dropbig').hidden=!v;if(v)$('dropbig').src=v.url;
 document.querySelectorAll('#dropviews figure').forEach(f=>f.classList.toggle('sel',!!v&&f.dataset.url===v.url));
 $('dropstandnote').textContent=v?'The dog stood here for this view (arrow), facing the dot.'
  :'No view picked: the place is saved at the dot.';plotMap(lastMap,lastPose)}
$('map').addEventListener('click',async e=>{if(!mapXf)return;const c=$('map'),b=c.getBoundingClientRect(),{s,ox,oy,m}=mapXf;
 const px=(e.clientX-b.left)*c.width/b.width,py=(e.clientY-b.top)*c.height/b.height;
 const x=m.origin[0]+(px-ox)/s*m.resolution,y=m.origin[1]+(m.height-(py-oy)/s)*m.resolution;
 if(x<m.origin[0]||y<m.origin[1]||x>m.origin[0]+m.width*m.resolution||y>m.origin[1]+m.height*m.resolution)return;
 DROP={x,y,view:null,spot:null};$('drop').hidden=false;$('dropxy').textContent=`${x.toFixed(2)}, ${y.toFixed(2)}`;
 $('dropspot').textContent='';$('dropviews').innerHTML='';$('dropvnote').textContent='Looking for camera views near here…';$('dropmsg').textContent='';pickView(null);
 const mine=DROP;let r;try{[r,COVER]=await Promise.all([fetch(`/trail/near?x=${x}&y=${y}`).then(q=>q.json()),fetch('/trail/coverage').then(q=>q.json())])}catch(err){$('dropvnote').textContent='Could not load camera views.';return}
 if(DROP!==mine)return;DROP.spot=r.spot;
 $('dropspot').innerHTML=r.spot==='obstacle'?'<span class="err">on furniture or a wall</span>':r.spot==='unseen'?'<span class="err">not mapped yet</span>':'open floor';
 const fv=r.views.filter(v=>v.facing).length;
 $('dropvnote').textContent=r.views.length?`${r.views.length} camera views from within 3 m${fv?`, ${fv} looking toward this spot`:''}. Pick one for the place's picture. Blue dots on the map show where views exist.`
  :'No camera views near here yet. They are kept while the dog drives around lined up with the map. You can still save the place, without a picture.';
 $('dropviews').innerHTML=r.views.map(v=>`<figure data-url="${v.url}"><img src="${v.url}" alt="camera view" loading="lazy"><figcaption>${hhmm(v.t)} · ${v.dist} m · ${v.facing?'spot '+side(v.bearing_deg):'facing away'}</figcaption></figure>`).join('');
 document.querySelectorAll('#dropviews figure').forEach((f,i)=>f.onclick=()=>pickView(r.views[i]));
 if(r.views[0]&&r.views[0].facing)pickView(r.views[0]);else plotMap(lastMap,lastPose);$('dropid').focus()});
$('dropx').addEventListener('click',()=>{DROP=null;$('drop').hidden=true;plotMap(lastMap,lastPose)});
$('dropform').addEventListener('submit',async e=>{e.preventDefault();if(!DROP)return;const v=DROP.view;
 const id=$('dropid').value.trim();
 if(PLACES[id]&&!confirm(`A place called ${id} already exists. Replace it?`))return;
 // the place: the thing is at the dot; the dog stands where it took the view, facing the dot
 const stand=v?{x:v.x,y:v.y,yaw_deg:Math.hypot(DROP.x-v.x,DROP.y-v.y)>0.3?Math.atan2(DROP.y-v.y,DROP.x-v.x)*180/Math.PI:v.yaw_deg}:null;
 const body={id,description:$('droppdesc').value,client:'console '+CLIENT,target:{x:DROP.x,y:DROP.y},stand,view:v?v.url:null};
 const r=await fetch('/places',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json().catch(()=>({}));
 if(!r.ok){$('dropmsg').innerHTML=`<span class="err">${esc(j.reason||r.status)}</span>`;return}
 $('dropid').value=$('droppdesc').value='';DROP=null;$('drop').hidden=true;await loadPlaces();plotMap(lastMap,lastPose);
 $('plnote').textContent=`saved ${id}`});
document.querySelectorAll('[data-map]').forEach(b=>b.addEventListener('click',async()=>{
 const ask={new:'Replace the map? Click again',clear:'Erase the map on screen? Click again',revert:'Drop unsaved changes? Click again'}[b.dataset.map];
 if(ask&&!b.dataset.sure){const t=b.textContent;b.dataset.sure=1;b.textContent=ask;setTimeout(()=>{delete b.dataset.sure;b.textContent=t},4000);return}
 if(b.dataset.sure){delete b.dataset.sure;b.textContent={new:'Start new map',clear:'Clear map',revert:'Back to saved map'}[b.dataset.map]}
 const r=await fetch('/map',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:b.dataset.map})});
 const j=await r.json().catch(()=>({}));$('mapstatus').textContent=r.ok?(j.status||'done'):(j.reason||('failed '+r.status))}));
</script></body></html>"""


def main():
    rclpy.init()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # rclpy otherwise swallows SIGTERM
    global NODE
    node = NODE = Collector()
    global MAPPER, TRAIL
    MAPPER = Mapper("/maps", os.environ.get("MAP_NAME", "lab"))
    TRAIL = Trail("/maps", os.environ.get("MAP_NAME", "lab"))
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print(f"Go2 dashboard on http://0.0.0.0:{PORT}  (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        REC.stop()
        MAPPER.autosave(latest["odom"])
        MAPPER.save_session(latest["odom"])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
