all: nbody.g nbody.i

nbody.g: nbody.c
	gcc -march=native -Ofast -mavx2 -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp -lcblas

nbody.i: nbody.c
	icc -xhost -Ofast -qopt-report $< -o $@  -qmkl -qopenmp -lm

clean:
	rm -Rf *~ nbody.g nbody.i *.optrpt

