import subprocess
import os
path = os.path.abspath(os.path.dirname(__file__))
for r in [500,1000,5000,10000,50000,100000,500000]:
    args = ["mpirun", "-c 6", "python", path + "/dds_parallel.py", str(r)]
    print(args)
    subprocess.run(args)
    exit(8)