#!/bin/bash
for SAMPLER in abc demcz dream fast fscabc lhs mc mcmc mle rope sa # sceua
do
    mpirun -c 6 python spotpy/examples/cli_cmf_lumped.py run -s $SAMPLER -n 10000 -p mpi
done
