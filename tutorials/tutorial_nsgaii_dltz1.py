import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

import spotpy
from spotpy.examples.spot_setup_dtlz1 import spot_setup

if __name__ == "__main__":
    # Create samplers for every algorithm:
    results = []
    n_obj = 3
    spot_setup = spot_setup(n_var=5, n_obj=n_obj)
    generations = 10
    n_pop = 30
    skip_duplicates = False

    sampler = spotpy.algorithms.NSGAII(
        spot_setup=spot_setup, dbname="NSGA2", dbformat="csv", save_sim=True
    )
    sampler.sample(generations, n_obj=3, n_pop=n_pop, skip_duplicates=skip_duplicates)

    last = None
    first = None

    # output calibration

    df = pd.read_csv("NSGA2.csv")

    df["like3"] = df.like3 * -1

    if last:
        df = df.iloc[-last:, :]
    elif first:
        df = df.iloc[:first, :]
    else:
        pass

    # plot objective functions
    fig = plt.figure()
    for i, name in enumerate(df.columns[:n_obj]):
        ax = fig.add_subplot(n_obj, 1, i + 1)
        df.loc[::5, name].plot(lw=0.5, figsize=(18, 8), ax=ax, color="black")
        plt.title(name)
    plt.show()

    x, y, z = df.iloc[-n_pop:, 0], df.iloc[-n_pop:, 1], df.iloc[-n_pop:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, marker="o")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    # plot parameters
    fig = plt.figure()
    for i, name in enumerate(df.columns[n_obj:8]):
        ax = fig.add_subplot(5, 1, i + 1)
        df.loc[:, name].plot(lw=0.5, figsize=(18, 8), ax=ax, color="black")
        plt.title(name)
    plt.show()
