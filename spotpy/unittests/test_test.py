import spotpy
import numpy as np
import spotpy.hymod.hymod
x = np.random.uniform(1.0, 100,5)

hyMod_sims = spotpy.hymod.hymod.hymod(x[0], x[1], x[2], x[3], x[4])
print(hyMod_sims)