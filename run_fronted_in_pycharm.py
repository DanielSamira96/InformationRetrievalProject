import json
# from mealpy.swarm_based.GWO import BaseGWO
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
interval = 30
startTime = datetime.now().strftime("%H_%M_%S")

os.mkdir(f'data/{startTime}')


def calculateMap(d):
    global index
    global max_value
    global d_max
    global x_axis
    global y_axis
    global interval
    d = list(d)
    requests.post(url + '/params', json=d)
    av_total = 0
    for q, true_wids in queries.items():
        try:
            res = requests.get(url + '/search', {'query': q}, timeout=35)
            if res.status_code == 200:
                pred_wids, _ = zip(*res.json())
                av_total += average_precision(true_wids, pred_wids)
        except:
            pass
    result = av_total / len(queries)
    print("params:" + str(d) + ", map:" + str(result))
    x_axis.append(index)
    y_axis.append(result)
    if result > max_value:
        max_value = result
        d_max = d
    if index % interval == 0:
        # Plot lists 'x' and 'y'
        plt.plot(x_axis, y_axis)

        # Plot axes labels and show the plot
        plt.xlabel('Iteration Index')
        plt.ylabel('Map@40')
        plt.xticks(range(0, index, math.ceil(index / 10)))
        # plt.show()
        plt.savefig(f'data/{startTime}/{str(index)}.png')
        print("Solution: [" + str(d_max) + "], globalMax: " + str(max_value))
    index += 1
    return result


def generate_position(lb=None, ub=None):
    """Start from a non-random position"""
    return np.array([0.5, 1.5, 2, 0.75])


problem_dict1 = {
    "fit_func": calculateMap,
    "lb": [0, 0, 0, 0],
    "ub": [1, 10, 10, 1],
    "minmax": "max"
    # "generate_position": generate_position
}

epoch = 20
pop_size = 20
neighbour_size = 10
model = OriginalHC(epoch, pop_size, neighbour_size)
best_position, best_fitness = model.solve(problem_dict1)
print(f"Solution: {best_position}, Fitness: {best_fitness}")

print("list_current_best:")
current_fit = []

for i, point in enumerate(model.history.list_current_best):
    current_fit.append(point[1][1][0])

numbers = range(1, len(current_fit) + 1)
numbers_2 = range(0, len(current_fit) + 1, 2)

figRos = plt.figure(figsize=(6, 6))
plt.plot(numbers, current_fit, marker="o", zorder=3)
plt.xticks(numbers_2)
plt.xlabel("Iteration Index")
plt.ylabel("Map@40")
# plt.show()
plt.savefig(f'data/{startTime}/list_current_best.png')

print("list_global_best:")
current_fit = []

for i, point in enumerate(model.history.list_global_best):
    current_fit.append(point[1][1][0])

numbers = range(1, len(current_fit) + 1)
numbers_2 = range(0, len(current_fit) + 1, 2)

figRos = plt.figure(figsize=(6, 6))
plt.plot(numbers, current_fit, marker="o", zorder=3)
plt.xticks(numbers_2)
plt.xlabel("Iteration Index")
plt.ylabel("Map@40")
# plt.show()
plt.savefig(f'data/{startTime}/list_global_best.png')
