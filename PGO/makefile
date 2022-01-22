
all : nbody.i nbody.g



nbody.g: nbody.c
	gcc -fprofile-generate -march=native -Ofast -mavx2 -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm -fopenmp -lcblas
	./nbody.g
	gcc -fprofile-use -march=native -Ofast -mavx2 -fopt-info-all=nbody.gcc.optrpt nbody.c -o $@ -lm -fopenmp -lcblas

nbody.i: nbody.c
	icc -prof-gen -qopt-report nbody.c -o $@ -qopenmp -ldl -xhost -Ofast -funroll-loops
	./nbody.i
	icc -prof-use -qopt-report $< -o $@ -qopenmp -ldl -xhost -Ofast -funroll-loops

clean:
	rm -Rf *~ nbody.g nbody.i nbody.gp nbody.ip  *.optrpt *.gcda *.dyn

