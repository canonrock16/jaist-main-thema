#!/usr/bin/env python
# coding: utf-8

import random
import time
from collections import defaultdict

import matplotlib.pyplot as pl
import neurokit2 as nk
import numpy as np
import pandas as pd
from brian2 import *
from smallworld import get_smallworld_graph
from smallworld.draw import draw_network
from tqdm import tqdm

start = time.time()
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
wmax = 10 * volt
Apre = 0.1 * volt
Apost = -0.12 * volt

n = 1000  # number of neurons in one group
neuron_group_count = 100
R = 0.8  # ratio about excitory-inhibitory neurons

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
    Pe.a = 0.02 / msecond
    Pe.b = 0.2 / msecond
    Pe.c = (15 * re ** 2 - 65) * mvolt
    Pe.d = (-6 * re ** 2 + 8) * mvolt / msecond
    Pe.I = 0.02 * volt / second

    Pi.a = (0.08 * ri ** 2 + 0.02) * 1 / msecond
    Pi.b = (-0.05 * ri ** 2 + 0.25) * 1 / msecond
    Pi.c = -65 * mvolt
    Pi.d = 2 * mvolt / msecond
    Pi.I = 0.02 * volt / second

    # グループ内の接続
    # 興奮性ニューロン to 同じニューロングループ内の100個のニューロンへのランダム接続
    Ce = Synapses(
        Pe,
        group,
        """
        w : volt
        dapre/dt = -apre/taupre : volt (event-driven)
        dapost/dt = -apost/taupost : volt (event-driven)
        """,
        on_pre="""
        v_post += w
        apre += Apre
        w = clip(w+apost, 0*volt, wmax)
        """,
        on_post="""
        apost += Apost
        w = clip(w+apre, 0*volt, wmax)
        """,
    )
    Ce.connect(p=0.1)
    Ce.w = 6.0 * volt
    Ce.delay = random.uniform(0, 20) * ms
    # 抑制性ニューロン　to 同じニューロングループ内の100個の興奮性ニューロンへのランダム接続
    Ci = Synapses(
        Pi,
        Pe,
        """
        w : volt
        dapre/dt = -apre/taupre : volt (event-driven)
        dapost/dt = -apost/taupost : volt (event-driven)
        """,
        on_pre="""
        v_post += w
        apre += Apre
        w = clip(w+apost, 0*volt, wmax)
        """,
        on_post="""
        apost += Apost
        w = clip(w+apre, 0*volt, wmax)
        """,
    )
    Ci.connect(p=0.125)
    Ci.w = -5.0 * volt
    Ci.delay = 1 * ms

    groups.append((group, Pe, Pi, Ce, Ci))

elapsed_time = time.time() - start
print("グループ内の配線をするまでにかかった時間", elapsed_time, "sec")
# WSモデルに従い、各グループの興奮性ニューロンから隣接する6つのノードへの接続と再配線を行う

# define network parameters
N = neuron_group_count
k_over_2 = 3
beta = 1.0
label = r"$\beta=0$"

focal_node = 0
# generate small-world graphs and draw
G = get_smallworld_graph(N, k_over_2, beta)

for edge in tqdm(list(G.edges())):
    source_group_Pe = groups[edge[0]][1]
    target_group = groups[edge[1]][0]

    Ce = Synapses(
        source_group_Pe,
        target_group,
        """
        w : volt
        dapre/dt = -apre/taupre : volt (event-driven)
        dapost/dt = -apost/taupost : volt (event-driven)
        """,
        on_pre="""
        v_post += w
        apre += Apre
        w = clip(w+apost, 0*volt, wmax)
        """,
        on_post="""
        apost += Apost
        w = clip(w+apre, 0*volt, wmax)
        """,
    )
    # 各興奮性ニューロンが、他のニューロングループと三本の接続を持つので、接続する確率は3/1000
    Ce.connect(p=0.003)
    Ce.w = 6.0 * volt
    Ce.delay = random.uniform(10, 30)

elapsed_time = time.time() - start
print("グループ外の配線をするまでにかかった時間", elapsed_time, "sec")

# とりあええず1000秒動かす
run_time_ms = 1000 * 1000

# statemonitorを1つだけにするver
value_interval_ms = 1
time_count = int(run_time_ms / value_interval_ms)

defaultclock.dt = value_interval_ms * ms
run(run_time_ms * ms)

elapsed_time = time.time() - start
print("最初の千秒までにかかった時間", elapsed_time, "sec")

# STDPの設定を外す
for group in groups:
    gruop[3].pre.code = "v_post +=w"
    gruop[4].pre.code = "v_post +=w"

# 100秒動かす
run_time_ms = 100 * 1000
run(run_time_ms * ms)

elapsed_time = time.time() - start
print("次の100秒までにかかった時間", elapsed_time, "sec")

# inputの設定を外す
new_I = array([0.0 for i in range(n * neuron_group_count)])
P.I = new_I * volt / second

# 100秒動かす
run_time_ms = 100 * 1000
V = StateMonitor(P, "v", record=True)
run(run_time_ms * ms)

elapsed_time = time.time() - start
print("次の100秒までにかかった時間", elapsed_time, "sec")


lap = defaultdict(list)
for i in tqdm(range(neuron_group_count)):
    start = int(i * n)
    end = int((i + 1) * n)

    group = V.v[start:end]
    # 興奮性
    Pe = group[: int(n * R)]
    for j in range(time_count):
        data = [neuron[j] / mV for neuron in list(group)]
        mean = sum(data) / len(data)
        lap[i].append(mean)

elapsed_time = time.time() - start
print("lap計算までにかかった時間", elapsed_time, "sec")


# 全てのニューロングループに対してMSEの計算を行う
results = []
for i in range(neuron_group_count):
    result = nk.entropy_multiscale(signal=np.array(lap[i]), scale=40, dimension=1)
    results.append(result[1]["Values"])

elapsed_time = time.time() - start
print("mse計算までにかかった時間", elapsed_time, "sec")

pd.DataFrame(results).to_csv("test.csv", index=False)
