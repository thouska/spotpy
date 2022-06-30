import numpy as np

import matplotlib.pylab as plt
import json
import matplotlib as mp

data_normalizer = mp.colors.Normalize()
color_map = mp.colors.LinearSegmentedColormap(
    "my_map",
    {
        "red": [(0, 1.0, 1.0),
                (1.0, .5, .5)],
        "green": [(0, 0.5, 0.5),
                  (1.0, 0, 0)],
        "blue": [(0, 0.50, 0.5),
                 (1.0, 0, 0)]
    }
)


def autolabel(ax, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%f' % height,
                ha='center', va='bottom')


def subplot(data, name, ylabel):
    fig = plt.figure(figsize=(20, 6))
    ax = plt.subplot(111)
    rep_labels = [str(j) for j in reps]
    x_pos = [i for i, _ in enumerate(rep_labels)]
    X = np.arange(len(data))
    ax_plot = ax.bar(x_pos, data, color=color_map(data_normalizer(data)), width=0.45)

    plt.xticks(x_pos, rep_labels)
    plt.xlabel("Repetitions")
    plt.ylabel(ylabel)

    autolabel(ax, ax_plot)
    plt.savefig(name + ".png")


parallel_data = json.loads('{"dds_duration": [1.1293659210205078, 3.254117250442505, 4.888171672821045, 18.719818592071533, 34.56907820701599, 169.47716689109802, 337.86882615089417, 1644.955144405365, 3348.948029756546], "rep": [30, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000], "dds_like": [-8384.884435178812, -8269.480874403698, -8268.453892284442, -8268.51195094138, -8269.65509041187, -8268.1421690868, -8267.791798085422, -8267.79178644684, -8268.141980514703]}')
reps = parallel_data["rep"]
subplot(parallel_data["dds_duration"], "DDS_PARALLEL_DURATION_all", "Duration of Run in Seconds")
subplot(parallel_data["dds_like"], "DDS_PARALLEL_OBJECTIVEFUNCTION_all", "Best Objective Function Value")
