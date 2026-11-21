#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_audio.py — 用 numpy 合成配乐，不依赖任何素材库。

思路和画面一致：**每一个音符也是算出来的**，不采样、不授权素材。

层次（全部 44.1k 立体声，加简易混响）：
  1. 低频铺底 drone —— D1/D2 的长音，带缓慢 LFO
  2. 和声 pad      —— 每镜换一次和弦，长起音
  3. 脉冲 pulse    —— 每 8 秒一次的低频轻击（像心跳，不是鼓点）
  4. 拨弦 arp      —— D 小调五声音阶，稀疏点缀
  5. 转场 whoosh   —— 每个镜头切换处，噪声扫频
  6. 章节 impact   —— 带章节号的镜头，低频下坠 + 噪声尾巴

用法:
    python make_audio.py out.wav --duration 117 [--scenes scenes.json]
"""
import argparse, json, wave
import numpy as np
from scipy.signal import lfilter

SR = 44100


def t_axis(dur):
    return np.arange(int(dur * SR)) / SR


def env_adsr(n, a, d, s_level, r):
    """简单包络：起音/衰减/保持/释放（秒 → 采样点）"""
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    s = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, s_level, d, endpoint=False),
        np.full(s, s_level),
        np.linspace(s_level, 0, r),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def tone(freq, dur, amp=1.0, detune=0.0, shape="sine"):
    t = t_axis(dur)
    f = freq * (1 + detune)
    if shape == "saw":
        # 带限近似：叠加前 8 次谐波，避免齿感过重
        w = sum(np.sin(2 * np.pi * f * k * t) / k for k in range(1, 9))
        return amp * w / 2.2
    return amp * np.sin(2 * np.pi * f * t)


def lowpass_fast(x, cutoff):
    """一阶低通（scipy lfilter，O(n)）"""
    a = float(np.exp(-2 * np.pi * cutoff / SR))
    return lfilter([1 - a], [1, -a], x)


def delay_fb(x, delay, g, feed=1.0):
    """y[n] = feed*x[n] + g*y[n-delay]"""
    a = np.zeros(delay + 1); a[0] = 1.0; a[delay] = -g
    b = np.zeros(delay + 1); b[0] = feed
    return lfilter(b, a, x)


def allpass(x, delay, g):
    """全通：b = [-g, 0.., 1], a = [1, 0.., -g]"""
    b = np.zeros(delay + 1); b[0] = -g; b[delay] = 1.0
    a = np.zeros(delay + 1); a[0] = 1.0; a[delay] = -g
    return lfilter(b, a, x)


def reverb(x, decay=0.35, mix=0.28):
    """Schroeder 混响：4 个梳状 + 2 个全通"""
    wet = (delay_fb(x, 1557, decay) + delay_fb(x, 1617, decay) +
           delay_fb(x, 1491, decay) + delay_fb(x, 1422, decay)) / 4
    wet = allpass(wet, 225, 0.5)
    wet = allpass(wet, 556, 0.5)
    return (1 - mix) * x + mix * wet


# ---------------------------------------------------------------- 作曲
SCALE = [0, 2, 3, 5, 7, 8, 10]          # 自然小调
ROOT = 73.42                            # D2


def deg(semi):
    return ROOT * (2 ** (semi / 12))


def build(duration, scenes, bpm=60):
    n = int(duration * SR)
    mixL = np.zeros(n, dtype=np.float64)
    mixR = np.zeros(n, dtype=np.float64)

    def add(sig, at, pan=0.0, gain=1.0):
        i = int(at * SR)
        if i >= n or i + len(sig) <= 0:
            return
        seg = sig[:max(0, n - i)]
        gl = gain * (0.5 - pan * 0.5) * 2
        gr = gain * (0.5 + pan * 0.5) * 2
        mixL[i:i + len(seg)] += seg * gl
        mixR[i:i + len(seg)] += seg * gr

    # 1) 铺底 drone：整片不断
    d = tone(ROOT / 2, duration, 0.26, detune=0.0) \
        + tone(ROOT, duration, 0.18, detune=0.0009) \
        + tone(ROOT * 1.5, duration, 0.09, detune=-0.0012)
    lfo = 0.72 + 0.28 * np.sin(2 * np.pi * 0.037 * t_axis(duration))
    d = d * lfo
    add(d, 0, -0.05, 0.9)

    # 2) 和声 pad：每镜一个和弦（取该镜前 85% 长度，留出切换空隙）
    chords = [[0, 3, 7], [0, 3, 7], [-2, 3, 5], [-5, 0, 3],
              [0, 3, 8], [0, 5, 8], [-3, 0, 5], [0, 3, 7]]
    for k, sc in enumerate(scenes):
        start = sc["start"] + 0.4
        dur = max(2.0, sc["dur"] * 0.9)
        ch = chords[k % len(chords)]
        sig = np.zeros(int(dur * SR))
        for semi in ch:
            sig += tone(deg(semi) * 2, dur, 0.13, detune=0.0007, shape="saw")
        sig = lowpass_fast(sig, 900)
        sig *= env_adsr(len(sig), 1.6, 1.2, 0.62, 2.2)
        add(sig, start, 0.0, 0.55)

    # 3) 脉冲：每 8 秒一次，很轻
    beat = 60.0 / bpm
    t = 0.0
    while t < duration:
        th = tone(55, 0.55, 0.5)
        th *= np.exp(-np.arange(len(th)) / (0.16 * SR))
        add(th, t, 0.0, 0.30)
        t += beat * 8

    # 4) 拨弦：稀疏点缀，固定种子保证可复现
    rng = np.random.default_rng(20261121)
    t = 2.0
    while t < duration:
        if rng.random() < 0.55:
            semi = SCALE[int(rng.integers(0, len(SCALE)))] + 12 * int(rng.integers(1, 3))
            pl = tone(deg(semi), 1.6, 0.32)
            pl *= np.exp(-np.arange(len(pl)) / (0.35 * SR))
            add(pl, t, float(rng.uniform(-0.4, 0.4)), 0.30)
        t += beat * 2

    # 5) 转场 whoosh + 6) 章节 impact
    for k, sc in enumerate(scenes):
        if k == 0:
            continue
        at = sc["start"]
        dur = 0.9
        m = int(dur * SR)
        noise = rng.normal(0, 1, m)
        sweep = np.linspace(0.0, 1.0, m) ** 2
        wh = noise * sweep * 0.16
        wh = lowpass_fast(wh, 2200)
        add(wh, at - dur * 0.75, float(np.sin(k)), 0.55)

        if sc.get("chapter"):
            boom = tone(58, 1.8, 0.55)
            boom *= np.exp(-np.arange(len(boom)) / (0.42 * SR))
            add(boom, at, 0.0, 0.55)
            tail = rng.normal(0, 1, int(1.2 * SR))
            tail *= np.exp(-np.arange(len(tail)) / (0.28 * SR)) * 0.10
            add(lowpass_fast(tail, 1400), at, 0.15, 0.5)

    # 混响 + 软限幅
    outL = reverb(mixL)
    outR = reverb(mixR)
    peak = max(np.abs(outL).max(), np.abs(outR).max(), 1e-9)
    outL = np.tanh(outL / peak * 1.3) * 0.50
    outR = np.tanh(outR / peak * 1.3) * 0.50
    return outL, outR


def write_wav(path, L, R):
    data = np.empty(len(L) * 2, dtype=np.int16)
    data[0::2] = np.clip(L * 32767, -32768, 32767).astype(np.int16)
    data[1::2] = np.clip(R * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--duration", type=float, default=117.0)
    ap.add_argument("--storyboard", default=None, help="可选：读 storyboard 的场景切点")
    a = ap.parse_args()

    if a.storyboard:
        sb = json.load(open(a.storyboard, encoding="utf-8"))
        scenes, acc = [], 0.0
        for s in sb["scenes"]:
            scenes.append({"start": acc, "dur": s["dur"], "chapter": s.get("chapter")})
            acc += s["dur"]
        dur = acc
    else:
        # 与《引力》一致的内置节奏
        plan = [(6, False), (16, True), (15, True), (15, True),
                (20, True), (20, True), (14, True), (11, False)]
        scenes, acc = [], 0.0
        for d, ch in plan:
            scenes.append({"start": acc, "dur": d, "chapter": ch})
            acc += d
        dur = acc

    L, R = build(dur, scenes)
    write_wav(a.out, L, R)
    print(f"OK  {a.out}  {dur:.1f}s  {len(L)/SR:.1f}s  {len(L)*4/1024/1024:.1f} MB")


if __name__ == "__main__":
    main()
