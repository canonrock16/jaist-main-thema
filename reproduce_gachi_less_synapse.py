#!/usr/bin/env python
# coding: utf-8

import time
from collections import defaultdict

# import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from brian2 import *
from smallworld import get_smallworld_graph
from smallworld.draw import draw_network
from tqdm import tqdm
import pickle
import os

import random

# スタンドアローンモードへ
set_device("cpp_standalone", build_on_run=False)
prefs.devices.cpp_standalone.openmp_threads = 128

time0 = time.time()
first_sim_ms = 1000 * 1000
second_sim_ms = 100 * 1000
third_sim_ms = 100 * 1000

n = 1000  # number of neurons in one group
neuron_group_count = 100

ws_model_beta = 1.0
timestamp = datetime.datetime.now().strftime("%Y–%m–%d_%H%M")

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
group                                            : integer (constant)
is_excitatory                                    : boolean (constant)
"""
)

reset = """
v = c
u = u + d
"""


taupre = taupost = 20 * ms
# 論文の設定
wmax = 10  # 本当に(ry
Apre = 0.1  # 本当にvolt?
Apost = -0.12  # 本当にvolt?

R = 0.8  # ratio about excitory-inhibitory neurons

# 各ニューロングループの生成
# 30mvでスパイクが発生する。数値積分法はeuler
P = NeuronGroup(n * neuron_group_count, model=eqs, threshold="v=>30*mvolt", reset=reset, method="euler")
re = np.random.random(int(n * neuron_group_count))
ri = np.random.random(round(n * neuron_group_count))

# TODO:exiteとinhibitのパラメータを各々変更する
P.group = "i // n"  # 0 for the first 1000 neurons, then 1 for the next 1000 neurons, etc.
P.is_excitatory = "(i % n) < int(R*n)"

# excitatory connections
P.a = 0.02 / msecond  # 正
P.b = 0.2 / msecond  # 正
P.c = (15 * re ** 2 - 65) * mvolt
P.d = (-6 * re ** 2 + 8) * mvolt / msecond
P.I = 20 * mvolt / msecond  # 他が全部mVなのにこれだけvoltはおかしい /msecondじゃないと頻度が低すぎる

# inhibitory connections
P.a["not is_excitatory"] = 0.1 * 1 / msecond
P.b["not is_excitatory"] = 0.2 * 1/msecond
P.c["not is_excitatory"] =  - 65*mvolt
P.d["not is_excitatory"] = 2 * mvolt / msecond
P.I["not is_excitatory"] = 20 * mvolt / msecond



# TODO:exiteとinhibitでclipの所を変える
Ce = Synapses(
        P,
        P,
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

Ci = Synapses(
        P,
        P,
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


# 興奮性ニューロン to 同じニューロングループ内の100個のニューロンへのランダム接続
Ce.connect("is_excitatory_pre and group_pre == group_post", p=0.1)

# 抑制性ニューロン　to 同じニューロングループ内の100個の興奮性ニューロンへのランダム接続
Ci.connect("not is_excitatory_pre and is_excitatory_post and group_pre == group_post", p=0.125)

# WSモデルに従い、各グループの興奮性ニューロンから隣接する6つのノードへの接続と再配線を行う
# define network parameters
N = neuron_group_count
k_over_2 = 3
# k_over_2 = 1
label = r"$\beta=0$"

focal_node = 0
# generate small-world graphs and draw
G = get_smallworld_graph(N, k_over_2, ws_model_beta)
inter_synapses = []

for edge in tqdm(list(G.edges())):
    source_group = edge[0]
    target_group = edge[1]
    # 各興奮性ニューロンが、他のニューロングループと三本の接続を持つので、接続する確率は3/1000
    Ce.connect(f"is_excitatory_pre and group_pre == {source_group} and group_post == {target_group}",p=0.003)

Ce.delay = 'rand()*20*ms'
seed()
Ci.delay["not is_excitatory_pre"] = 1 * ms

Ce.w = 6.0
Ci.w["not is_excitatory_pre"] = -5.0

time1 = time.time()
print("配線をするまでにかかった時間", time1 - time0, "sec")

net = Network(P,Ce,Ci)

time2 = time.time()
# とりあええず1000秒動かす
value_interval_ms = 1
defaultclock.dt = value_interval_ms * ms
net.run(first_sim_ms * ms, report="stdout")

time3 = time.time()
print("最初の千秒までにかかった時間", time3 - time2, "sec")


# STDPの設定を外す
Ce.pre.code = "v_post +=w* mV"
Ci.post.code = ""
    
# 100秒動かす
net.run(second_sim_ms * ms, report="stdout")

time4 = time.time()
print("次の100秒までにかかった時間", time4 - time3, "sec")


# inputの設定を外す
new_I = array([0.0 for i in range(n * neuron_group_count)])
P.I = new_I * volt / second

V = StateMonitor(P, "v", record=True)
S = SpikeMonitor(P)

net.add(V)
net.add(S)

# 100秒動かす
net.run(third_sim_ms * ms, report="stdout")


time5 = time.time()
print("次の100秒までにかかった時間", time5 - time4, "sec")

# スタンドアローンモードへ
device.build(directory=f"output_{ws_model_beta}", compile=True, run=True, debug=False)

# wを保存
savedir = f"./results/pickle/{timestamp}"
os.makedirs(savedir, exist_ok=True)
with open(f"{savedir}/w_Ce_{ws_model_beta}.pkl", "wb") as f:
    pickle.dump(np.array(Ce.w), f)
with open(f"{savedir}/w_Ci_{ws_model_beta}.pkl", "wb") as f:
    pickle.dump(np.array(Ci.w), f)

    
# スパイクを保存
spikes = S.spike_trains()
with open(f"{savedir}/spikes_{ws_model_beta}.pkl", "wb") as f:
    pickle.dump(spikes, f)

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

savedir = f"./results/csv/{timestamp}"
os.makedirs(savedir,exist_ok=True)
pd.DataFrame(lap[0]).to_csv(f"{savedir}/test_before_{ws_model_beta}.csv", index=False)


# 全てのニューロングループに対してMSEの計算を行う
results = []
for i in range(neuron_group_count):
    result = nk.entropy_multiscale(signal=np.array(lap[i]), scale=40, dimension=1)
    results.append(result[1]["Values"])

time7 = time.time()
print("mse計算までにかかった時間", time7 - time6, "sec")

pd.DataFrame(results).to_csv(f"{savedir}/test_{ws_model_beta}.csv", index=False)
time8 = time.time()
print("全体の処理時間", time8 - time0, "sec")
