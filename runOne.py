import json
# from mealpy.swarm_based.GWO import BaseGWO
import random

from mealpy.math_based.HC import OriginalHC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcloros
import requests
from time import time
import math
from datetime import datetime
import os

with open('queries_train.json', 'rt') as f:
    queries = json.load(f)


def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


def recall(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i, doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions) + 1) / (i + 1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions) / len(precisions), 3)


url = 'http://34.122.57.217:8080'

global index
global max_value
global d_max
global x_axis
global y_axis
global interval
global startTime
index = 1
x_axis = []
y_axis = []
epoch = 3
max_value = 0
d_max = []
interval = 10
startTime = datetime.now().strftime("%H_%M_%S")



# d = [0.5611646, 4.2845903, 8.8012513, 0.3356693]
d = [0.8, 7.1748497319651925, 10.0, 0.0, 0.4021003780164265, 4.793785221191588, 0.7444968616996552, 0.1]
# d = [0.7400346359741982, 7.1748497319651925, 10.0, 0.0, 0.4021003780164265, 4.793785221191588, 0.7444968616996552, 0.1805913789221115]
# d = [0.3944550]
requests.post(url + '/params', json=d)
av_total = 0
duration_total = 0
qs_res = []
duration_list = []
map_avg = []
for q, true_wids in queries.items():
    duration, ap = None, None
    t_start = time()
    try:
        res = requests.get(url + '/search', {'query': q}, timeout=35)
        duration = time() - t_start
        duration_total += duration
        if res.status_code == 200:
            pred_wids, _ = zip(*res.json())
            ap = average_precision(true_wids, pred_wids)
            av_total += ap
    except:
        pass
    duration_list.append(duration)
    map_avg.append(ap)
    qs_res.append((q, duration, ap))

for i in range(len(map_avg)):
    # list_point = list(point[0])
    plt.plot(duration_list[i], map_avg[i], marker="o", markersize=8, markerfacecolor=mcloros.cnames.get(i),
             zorder=3, alpha=0.5)
# plt.show()
print("Average av = " + str(av_total / len(qs_res)))
print("Average duration = " + str(duration_total / len(qs_res)))
for row in qs_res:
    print(row)
plt.xlabel('Duration time')
plt.ylabel('Map@40')
plt.savefig(f'data/BM25TITLEONLYquerieswithbestparam.png')
index += 1
