#!/bin/bash

# works with Linux

echo "This scripts works best with Linux / Ubuntu"
echo "cython needs to be installed (with pip install cython)"

cython --cplus -3 hymod.pyx 
g++ hymod.cpp main_hymod.cpp -o hymod $(python3-config --includes) $(python3-config --cflags) $(python3-config --ldflags) -fPIC
