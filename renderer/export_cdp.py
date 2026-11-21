#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_cdp.py — 单会话 CDP 逐帧渲染（最快的一条路）。

启动一次无头 Edge，通过 DevTools 协议反复 seek + 截图，
每帧约 0.1–0.3 秒，两分钟片子十分钟内跑完。

用法:
  python export_cdp.py --duration 117 --fps 30 --out gravity.mp4
  python export_cdp.py --duration 117 --fps 30 --limit 30   # 只渲前 30 帧
"""
import argparse, base64, json, os, shutil, subprocess, sys, tempfile, time, glob
import urllib.request
import cv2
import websocket

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
FW, FH = 1920, 1080
PORT = 9300 + (os.getpid() % 500)


def wait_port(url, tries=120):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                return json.load(r)
        except Exception:
            time.sleep(0.25)
    return None


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=60,
                                              max_size=64 * 1024 * 1024)
        self.id = 0

    def send(self, method, params=None):
        self.id += 1
        self.ws.send(json.dumps({"id": self.id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=os.path.join(os.path.dirname(__file__), "..", "gravity.html"))
    ap.add_argument("--duration", type=float, default=117)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "gravity_v1.mp4"))
    ap.add_argument("--frames", default=os.path.join(os.path.dirname(__file__), "..", "frames"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--encode-only", action="store_true")
    a = ap.parse_args()

    fdir = os.path.abspath(a.frames)
    os.makedirs(fdir, exist_ok=True)
    n_total = int(round(a.duration * a.fps))
    i0 = int(round(a.start * a.fps))

    html = os.path.abspath(a.html)
    url = "file:///" + html.replace("\\", "/") + "?export=1"

    prof = tempfile.mkdtemp(prefix="edgecdp_")
    proc = None
    ok = 0

    if not a.encode_only:
        proc = subprocess.Popen(
            [EDGE, "--headless=new", "--disable-gpu", "--no-sandbox",
             "--hide-scrollbars", "--no-first-run", "--disable-extensions",
             "--disable-background-networking", "--mute-audio",
             f"--remote-debugging-port={PORT}", "--remote-allow-origins=*",
             f"--user-data-dir={prof}",
             f"--window-size={FW},{FH}",
             "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        ver = wait_port(f"http://127.0.0.1:{PORT}/json/version")
        if not ver:
            sys.exit("浏览器没起来")

        target = wait_port(f"http://127.0.0.1:{PORT}/json/list")
        page = None
        for t in target or []:
            if t.get("type") == "page":
                page = t
                break
        if page is None:
            sys.exit("没有页面目标")

        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Emulation.setDeviceMetricsOverride",
                 {"width": FW, "height": FH, "deviceScaleFactor": 1, "mobile": False})
        cdp.send("Page.navigate", {"url": url})
        time.sleep(2.5)

        # 等 film 就绪
        for _ in range(40):
            r = cdp.send("Runtime.evaluate",
                         {"expression": "!!(window.__film && window.__film.seek)",
                          "returnByValue": True})
            if r.get("result", {}).get("value"):
                break
            time.sleep(0.25)

        t0 = time.time()
        for i in range(i0, min(n_total, i0 + (a.limit or n_total))):
            p = os.path.join(fdir, f"{i:05d}.jpg")
            if os.path.exists(p):
                ok += 1
                continue
            cdp.send("Runtime.evaluate",
                     {"expression": f"window.__film.seek({i / a.fps:.5f})",
                      "returnByValue": True})
            shot = cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 92})
            open(p, "wb").write(base64.b64decode(shot["data"]))
            ok += 1
            if (i + 1) % 200 == 0:
                dt = time.time() - t0
                print(f"  {i+1}/{n_total}  {dt/(i+1)*1000:.0f} ms/帧", flush=True)

        print(f"渲染完成 {ok} 帧，用时 {time.time()-t0:.0f}s", flush=True)
        cdp.close()
        proc.terminate()
        try: proc.wait(timeout=10)
        except Exception: proc.kill()
        shutil.rmtree(prof, ignore_errors=True)

    have = sorted(glob.glob(os.path.join(fdir, "*.jpg")) + glob.glob(os.path.join(fdir, "*.png")))
    if not have:
        sys.exit("没有可用帧")
    print(f"编码 {len(have)} 帧 → {a.out}", flush=True)
    out = os.path.abspath(a.out)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (FW, FH))
    for k, p in enumerate(have):
        img = cv2.imread(p)
        if img is None:
            continue
        if img.shape[:2] != (FH, FW):
            img = cv2.resize(img, (FW, FH))
        vw.write(img)
    vw.release()
    print(f"完成：{out}  {os.path.getsize(out)/1024/1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
