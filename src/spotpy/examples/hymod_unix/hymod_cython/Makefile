all: prepare hymod

prepare:
	@echo "This scripts works best with Linux / Ubuntu"
	@echo "cython needs to be installed (with pip install cython or make cython)"
	cython --cplus -3 hymod.pyx 

cython:
	@echo pip i cython
	pip install cython

hymod:
	g++ hymod.cpp main_hymod.cpp -o hymod `python3-config --includes` `python3-config --cflags` `python3-config --ldflags` -fPIC

clean:
	rm -r hymod.dSYM hymod.cpp hymod.h 

run:
	./hymod