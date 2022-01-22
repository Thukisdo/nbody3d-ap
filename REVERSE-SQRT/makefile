all: nbody.g nbody.i

nbody.g: nbody.c
	gcc -march=native -Ofast -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp -lcblas

nbody.i: nbody.c
	icc -qopt-report $< -o $@ -qopenmp -ldl -xhost -Ofast -funroll-loops

clean:
	rm -Rf *~ nbody.g nbody.i *.optrpt

