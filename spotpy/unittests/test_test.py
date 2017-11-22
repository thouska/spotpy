import spotpy
x = [spotpy.parameter.Uniform('x1', low=1.0, high=500, optguess=412.33),
                       spotpy.parameter.Uniform('x2', low=0.1, high=2.0, optguess=0.1725),
                       spotpy.parameter.Uniform('x3', low=0.1, high=0.99, optguess=0.8127),
                       spotpy.parameter.Uniform('x4', low=0.0, high=0.10, optguess=0.0404),
                       spotpy.parameter.Uniform('x5', low=0.1, high=0.99, optguess=0.5592)
                       ]

hyMod_sims = spotpy.hymod.hymod.hymod(x[0], x[1], x[2], x[3], x[4])
print(hyMod_sims)