#!/usr/bin/env python3
"""
SysMonitor - Linux System Monitor
Real-time CPU, GPU, Memory, Disk, Network monitoring with process control
"""

import json
import time
import os
import signal
import subprocess
import threading
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    import psutil
except ImportError:
    os.system("pip install psutil --break-system-packages -q")
    import psutil

try:
    from flask_cors import CORS
except ImportError:
    os.system("pip install flask-cors --break-system-packages -q")
    from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

# ─── GPU Detection ────────────────────────────────────────────────────────────


def get_gpu_info():
    gpus = []
    # Try NVIDIA
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.graphics",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split("\n")):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    gpus.append(
                        {
                            "index": i,
                            "vendor": "NVIDIA",
                            "name": parts[0] if len(parts) > 0 else "Unknown",
                            "utilization": (
                                float(parts[1])
                                if len(parts) > 1 and parts[1] != "[N/A]"
                                else 0
                            ),
                            "mem_used": (
                                float(parts[2])
                                if len(parts) > 2 and parts[2] != "[N/A]"
                                else 0
                            ),
                            "mem_total": (
                                float(parts[3])
                                if len(parts) > 3 and parts[3] != "[N/A]"
                                else 0
                            ),
                            "temperature": (
                                float(parts[4])
                                if len(parts) > 4 and parts[4] != "[N/A]"
                                else 0
                            ),
                            "power": (
                                float(parts[5])
                                if len(parts) > 5 and parts[5] not in ("[N/A]", "N/A")
                                else 0
                            ),
                            "clock": (
                                float(parts[6])
                                if len(parts) > 6 and parts[6] not in ("[N/A]", "N/A")
                                else 0
                            ),
                        }
                    )
            if gpus:
                return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try AMD (rocm-smi)
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--json"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for i, (key, val) in enumerate(data.items()):
                if key.startswith("card"):
                    gpus.append(
                        {
                            "index": i,
                            "vendor": "AMD",
                            "name": val.get("Card series", "AMD GPU"),
                            "utilization": float(val.get("GPU use (%)", 0)),
                            "mem_used": float(val.get("GPU memory use (%)", 0)),
                            "mem_total": 100,
                            "temperature": float(
                                str(
                                    val.get("Temperature (Sensor edge) (C)", 0)
                                ).replace("c", "")
                            ),
                            "power": 0,
                            "clock": 0,
                        }
                    )
            if gpus:
                return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # No GPU found — return simulated for demo
    return [
        {
            "index": 0,
            "vendor": "N/A",
            "name": "No GPU Detected",
            "utilization": 0,
            "mem_used": 0,
            "mem_total": 0,
            "temperature": 0,
            "power": 0,
            "clock": 0,
        }
    ]


# ─── CPU Temperature ──────────────────────────────────────────────────────────


def get_cpu_temps():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return []
        # Prefer coretemp / k10temp
        for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
            if key in temps:
                return [
                    {
                        "label": t.label or f"Core {i}",
                        "current": t.current,
                        "high": t.high,
                        "critical": t.critical,
                    }
                    for i, t in enumerate(temps[key])
                ]
        # Fallback: first available
        first = next(iter(temps.values()))
        return [
            {
                "label": t.label or f"Sensor {i}",
                "current": t.current,
                "high": t.high,
                "critical": t.critical,
            }
            for i, t in enumerate(first)
        ]
    except Exception:
        return []


# ─── System Stats ─────────────────────────────────────────────────────────────

_prev_net = psutil.net_io_counters()
_prev_net_time = time.time()
_prev_disk = psutil.disk_io_counters()
_prev_disk_time = time.time()


def get_stats():
    global _prev_net, _prev_net_time, _prev_disk, _prev_disk_time

    now = time.time()

    # CPU
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    cpu_freq = psutil.cpu_freq(percpu=False)
    cpu_freq_per = psutil.cpu_freq(percpu=True)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_times = psutil.cpu_times_percent(interval=None)
    load_avg = os.getloadavg() if hasattr(os, "getloadavg") else [0, 0, 0]

    # Memory
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # Disk
    disk_root = psutil.disk_usage("/")
    disk_io = psutil.disk_io_counters()
    disk_dt = now - _prev_disk_time
    disk_read_speed = (
        (disk_io.read_bytes - _prev_disk.read_bytes) / disk_dt if disk_dt > 0 else 0
    )
    disk_write_speed = (
        (disk_io.write_bytes - _prev_disk.write_bytes) / disk_dt if disk_dt > 0 else 0
    )
    _prev_disk = disk_io
    _prev_disk_time = now

    # Network
    net_io = psutil.net_io_counters()
    net_dt = now - _prev_net_time
    net_sent_speed = (
        (net_io.bytes_sent - _prev_net.bytes_sent) / net_dt if net_dt > 0 else 0
    )
    net_recv_speed = (
        (net_io.bytes_recv - _prev_net.bytes_recv) / net_dt if net_dt > 0 else 0
    )
    _prev_net = net_io
    _prev_net_time = now

    # Uptime
    boot_time = psutil.boot_time()
    uptime_sec = int(now - boot_time)

    return {
        "timestamp": now,
        "uptime": uptime_sec,
        "cpu": {
            "percent": cpu_percent,
            "per_core": cpu_per_core,
            "count_logical": cpu_count_logical,
            "count_physical": cpu_count_physical,
            "freq_mhz": cpu_freq.current if cpu_freq else 0,
            "freq_max": cpu_freq.max if cpu_freq else 0,
            "freq_per_core": [f.current for f in cpu_freq_per] if cpu_freq_per else [],
            "user": cpu_times.user,
            "system": cpu_times.system,
            "iowait": getattr(cpu_times, "iowait", 0),
            "idle": cpu_times.idle,
            "load_avg": list(load_avg),
            "temps": get_cpu_temps(),
        },
        "memory": {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percent": mem.percent,
            "buffers": getattr(mem, "buffers", 0),
            "cached": getattr(mem, "cached", 0),
            "swap_total": swap.total,
            "swap_used": swap.used,
            "swap_percent": swap.percent,
        },
        "disk": {
            "total": disk_root.total,
            "used": disk_root.used,
            "free": disk_root.free,
            "percent": disk_root.percent,
            "read_speed": disk_read_speed,
            "write_speed": disk_write_speed,
            "read_count": disk_io.read_count,
            "write_count": disk_io.write_count,
        },
        "network": {
            "sent_speed": net_sent_speed,
            "recv_speed": net_recv_speed,
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        },
        "gpu": get_gpu_info(),
    }


# ─── Process List ─────────────────────────────────────────────────────────────


def get_processes(sort_by="cpu", limit=50):
    procs = []
    attrs = [
        "pid",
        "name",
        "username",
        "status",
        "cpu_percent",
        "memory_percent",
        "memory_info",
        "num_threads",
        "create_time",
        "nice",
        "cmdline",
    ]
    for proc in psutil.process_iter(attrs):
        try:
            info = proc.info
            procs.append(
                {
                    "pid": info["pid"],
                    "name": info["name"] or "",
                    "user": info["username"] or "",
                    "status": info["status"] or "",
                    "cpu": round(info["cpu_percent"] or 0, 1),
                    "mem_percent": round(info["memory_percent"] or 0, 2),
                    "mem_rss": info["memory_info"].rss if info["memory_info"] else 0,
                    "threads": info["num_threads"] or 0,
                    "nice": info["nice"] or 0,
                    "started": info["create_time"] or 0,
                    "cmd": " ".join(info["cmdline"] or [])[:80],
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    key = "cpu" if sort_by == "cpu" else "mem_percent"
    procs.sort(key=lambda x: x[key], reverse=True)
    return procs[:limit]


# ─── API Routes ───────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/api/processes")
def api_processes():
    sort_by = request.args.get("sort", "cpu")
    limit = int(request.args.get("limit", 50))
    return jsonify(get_processes(sort_by, limit))


@app.route("/api/process/<int:pid>/kill", methods=["POST"])
def api_kill(pid):
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        proc.kill()
        return jsonify({"ok": True, "msg": f"Killed {name} (PID {pid})"})
    except psutil.NoSuchProcess:
        return jsonify({"ok": False, "msg": f"PID {pid} not found"}), 404
    except psutil.AccessDenied:
        return jsonify({"ok": False, "msg": f"Permission denied (try sudo)"}), 403
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/process/<int:pid>/terminate", methods=["POST"])
def api_terminate(pid):
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        proc.terminate()
        return jsonify({"ok": True, "msg": f"Terminated {name} (PID {pid})"})
    except psutil.NoSuchProcess:
        return jsonify({"ok": False, "msg": f"PID {pid} not found"}), 404
    except psutil.AccessDenied:
        return jsonify({"ok": False, "msg": f"Permission denied"}), 403
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/process/<int:pid>/suspend", methods=["POST"])
def api_suspend(pid):
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        proc.suspend()
        return jsonify({"ok": True, "msg": f"Suspended {name} (PID {pid})"})
    except psutil.NoSuchProcess:
        return jsonify({"ok": False, "msg": f"PID {pid} not found"}), 404
    except psutil.AccessDenied:
        return jsonify({"ok": False, "msg": f"Permission denied"}), 403
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/process/<int:pid>/resume", methods=["POST"])
def api_resume(pid):
    try:
        proc = psutil.Process(pid)
        name = proc.name()
        proc.resume()
        return jsonify({"ok": True, "msg": f"Resumed {name} (PID {pid})"})
    except psutil.NoSuchProcess:
        return jsonify({"ok": False, "msg": f"PID {pid} not found"}), 404
    except psutil.AccessDenied:
        return jsonify({"ok": False, "msg": f"Permission denied"}), 403
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/process/<int:pid>/nice", methods=["POST"])
def api_nice(pid):
    data = request.get_json() or {}
    nice_val = int(data.get("nice", 0))
    try:
        proc = psutil.Process(pid)
        proc.nice(nice_val)
        return jsonify({"ok": True, "msg": f"Set nice={nice_val} for PID {pid}"})
    except psutil.AccessDenied:
        return jsonify({"ok": False, "msg": "Permission denied"}), 403
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/api/disks")
def api_disks():
    disks = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
            disks.append(
                {
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "fstype": part.fstype,
                    "total": usage.total,
                    "used": usage.used,
                    "free": usage.free,
                    "percent": usage.percent,
                }
            )
        except Exception:
            pass
    return jsonify(disks)


@app.route("/api/network/interfaces")
def api_net_interfaces():
    ifaces = []
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()
    for name, addr_list in addrs.items():
        iface_stat = stats.get(name)
        ifaces.append(
            {
                "name": name,
                "isup": iface_stat.isup if iface_stat else False,
                "speed": iface_stat.speed if iface_stat else 0,
                "addresses": [
                    {"family": str(a.family), "address": a.address} for a in addr_list
                ],
            }
        )
    return jsonify(ifaces)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7070
    print(f"\n{'=' * 50}")
    print(f"  SysMonitor running at http://localhost:{port}")
    print(f"{'=' * 50}\n")
    # Pre-warm cpu_percent
    psutil.cpu_percent(interval=0.1)
    psutil.cpu_percent(interval=None, percpu=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
