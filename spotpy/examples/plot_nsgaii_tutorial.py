import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



# user config

n_var = 5
n_obj = 3

last = None
first = 2000

# output calibration 

df = pd.read_csv("NSGA2.csv")

if last:
    df = df.iloc[-last:,:]
elif first:
    df = df.iloc[:first,:]
else:
    pass



print(len(df))
# plot objective functions
fig = plt.figure()
for i,name in enumerate(df.columns[:n_obj]):
    ax = fig.add_subplot(n_obj,1,i +1)
    df.loc[::5,name].plot(lw=0.5,figsize=(18,8),ax = ax,color="black")
    plt.title(name)
plt.show()

x,y,z = df.iloc[:,0],df.iloc[:,1],df.iloc[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,marker="o")
ax.set_xlabel("f0")
ax.set_ylabel("f1")
ax.set_zlabel("f2")
plt.show()

# plot parameters
fig = plt.figure()
for i,name in enumerate(df.columns[n_obj:8]):
    ax = fig.add_subplot(5,1,i +1)
    df.loc[:,name].plot(lw=0.5,figsize=(18,8),ax = ax,color="black")
    plt.title(name)
plt.show()



