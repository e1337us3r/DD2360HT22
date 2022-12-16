ARCH=sm_86

build:
	nvcc -arch=$(ARCH) -O3 -I../common lab3_ex1.cu -o ex.o

plot:
	python3 plot.py

clean:
	rm *.o
