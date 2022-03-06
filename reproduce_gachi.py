#!/usr/bin/env python
# coding: utf-8

import time
from collections import defaultdict

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from brian2 import *
from smallworld import get_smallworld_graph
from smallworld.draw import draw_network
from tqdm import tqdm

import random

# スタンドアローンモードへ
set_device("cpp_standalone", build_on_run=False)
prefs.devices.cpp_standalone.openmp_threads = 64

time0 = time.time()
# イジケビッチニューロンの定義
eqs = Equations(
    """
dv/dt = (0.04/ms/mV)*v**2+(5/ms)*v+140*mV/ms-u+I : volt
du/dt = a*(b*v-u)                                : volt/second
I                                                : volt/second
a                                                : 1/second
b                                                : 1/second
c                                                : volt
d                                                : volt/second
"""
)

reset = """
v = c
u = u + d
"""


taupre = taupost = 20 * ms
# 論文の設定
wmax = 10
Apre = 0.1
Apost = -0.12

n = 1000  # number of neurons in one group
neuron_group_count = 100
R = 0.8  # ratio about excitory-inhibitory neurons
first_sim_ms = 1000 * 1000
second_sim_ms = 100 * 1000
third_sim_ms = 100 * 1000

# 各ニューロングループの生成
# 30mvでスパイクが発生する。数値積分法はeuler
P = NeuronGroup(n * neuron_group_count, model=eqs, threshold="v>30*mvolt", reset=reset, method="euler")
re = np.random.random(int(n * R))
ri = np.random.random(round(n * (1 - R)))

groups = []
# サブグループに分ける
for i in tqdm(range(neuron_group_count)):
    start = int(i * n)
    end = int((i + 1) * n)

    group = P[start:end]
    # 興奮性
    Pe = group[: int(n * R)]
    # 抑制性
    Pi = group[int(n * R) :]

    # 各種設定
    Pe.a = 0.02 / msecond  # 正
    Pe.b = 0.2 / msecond  # 正
    Pe.c = (15 * re ** 2 - 65) * mvolt
    Pe.d = (-6 * re ** 2 + 8) * mvolt / msecond
    Pe.I = 20 * mvolt / msecond  # 他が全部mVなのにこれだけvoltはおかしい /msecondじゃないと頻度が低すぎる
    # Pe.u = Pe.b * Pe.c
    # Pe.v = Pe.c

    Pi.a = (0.08 * ri ** 2 + 0.02) * 1 / msecond  # 正
    Pi.b = (-0.05 * ri ** 2 + 0.25) * 1 / msecond  # 正
    Pi.c = -65 * mvolt
    Pi.d = 2 * mvolt / msecond
    Pi.I = 20 * mvolt / msecond
    # Pi.u = Pi.b * Pi.c
    # Pi.v = Pi.c

    # グループ内の接続
    # 興奮性ニューロン to 同じニューロングループ内の100個のニューロンへのランダム接続
    Ce = Synapses(
        Pe,
        group,
        """
        w : 1
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        """,
        on_pre="""
        v_post += w * mV
        apre += Apre
        w = clip(w+apost, 0, wmax)
        """,
        on_post="""
        apost += Apost
        w = clip(w+apre, 0, wmax)
        """,
    )
    Ce.connect(p=0.1)
    Ce.w = 6.0
    Ce.delay = round(random.uniform(0, 20), 2) * ms
    # 抑制性ニューロン　to 同じニューロングループ内の100個の興奮性ニューロンへのランダム接続
    Ci = Synapses(
        Pi,
        Pe,
        """
        w : 1
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        """,
        on_pre="""
        v_post += w * mV
        apre += Apre
        w = clip(w+apost, -wmax, 0)
        """,
        on_post="""
        apost += Apost
        w = clip(w+apre, -wmax, 0)
        """,
    )
    Ci.connect(p=0.125)
    Ci.w = -5.0
    Ci.delay = 1 * ms

    groups.append((group, Pe, Pi, Ce, Ci))

time1 = time.time()
print("グループ内の配線をするまでにかかった時間", time1 - time0, "sec")

# WSモデルに従い、各グループの興奮性ニューロンから隣接する6つのノードへの接続と再配線を行う

# define network parameters
N = neuron_group_count
k_over_2 = 3
beta = 1.0
label = r"$\beta=0$"

focal_node = 0
# generate small-world graphs and draw
G = get_smallworld_graph(N, k_over_2, beta)
inter_synapses = []

for edge in tqdm(list(G.edges())):
    source_group_Pe = groups[edge[0]][1]
    target_group = groups[edge[1]][0]

    Ce = Synapses(
        source_group_Pe,
        target_group,
        """
        w : 1
        dapre/dt = -apre/taupre : 1 (event-driven)
        dapost/dt = -apost/taupost : 1 (event-driven)
        """,
        on_pre="""
        v_post += w *mV
        apre += Apre
        w = clip(w+apost, 0, wmax)
        """,
        on_post="""
        apost += Apost
        w = clip(w+apre, 0, wmax)
        """,
    )
    # 各興奮性ニューロンが、他のニューロングループと三本の接続を持つので、接続する確率は3/1000
    Ce.connect(p=0.003)
    Ce.w = 6.0
    Ce.delay = round(random.uniform(10, 30), 2) * ms
    inter_synapses.append(Ce)

time2 = time.time()
print("グループ外の配線をするまでにかかった時間", time2 - time1, "sec")


net = Network([group[3] for group in groups], [group[4] for group in groups], inter_synapses, P)

# とりあええず1000秒動かす
value_interval_ms = 1
defaultclock.dt = value_interval_ms * ms
net.run(first_sim_ms * ms, report="stdout")

time3 = time.time()
print("最初の千秒までにかかった時間", time3 - time2, "sec")


# STDPの設定を外す
for group in groups:
    group[3].pre.code = "v_post +=w* mV"
    group[3].post.code = ""
    group[4].pre.code = "v_post +=w *mV"
    group[4].post.code = ""

for inter in inter_synapses:
    inter.pre.code = "v_post +=w *mV"
    inter.post.code = ""

# 100秒動かす
net.run(second_sim_ms * ms, report="stdout")

time4 = time.time()
print("次の100秒までにかかった時間", time4 - time3, "sec")


# inputの設定を外す
new_I = array([0.0 for i in range(n * neuron_group_count)])
P.I = new_I * volt / second

V = StateMonitor(P, "v", record=True)
A = StateMonitor(P, "a", record=True)
B = StateMonitor(P, "b", record=True)
C = StateMonitor(P, "c", record=True)
D = StateMonitor(P, "d", record=True)
U = StateMonitor(P, "u", record=True)
S = SpikeMonitor(P)

net.add(V)
net.add(A)
net.add(B)
net.add(C)
net.add(D)
net.add(U)
net.add(S)

# 100秒動かす
net.run(third_sim_ms * ms, report="stdout")


time5 = time.time()
print("次の100秒までにかかった時間", time5 - time4, "sec")

# スタンドアローンモードへ
device.build(directory="output", compile=True, run=False, debug=False)

lap = defaultdict(list)
for i in tqdm(range(neuron_group_count)):
    start = int(i * n)
    end = int((i + 1) * n)

    group = V.v[start:end] / mV
    # 興奮性
    Pe = group[: int(n * R)]
    lap[i] = np.mean(np.array(Pe), axis=0)

time6 = time.time()
print("lap計算までにかかった時間", time6 - time5, "sec")

pd.DataFrame(lap[0]).to_csv("test_before.csv", index=False)


# 全てのニューロングループに対してMSEの計算を行う
results = []
for i in range(neuron_group_count):
    result = nk.entropy_multiscale(signal=np.array(lap[i]), scale=40, dimension=1)
    results.append(result[1]["Values"])

time7 = time.time()
print("mse計算までにかかった時間", time7 - time6, "sec")

pd.DataFrame(results).to_csv("test.csv", index=False)
time8 = time.time()
print("全体の処理時間", time8 - time0, "sec")
