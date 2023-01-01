ARCH=sm_86

build:
	nvcc -arch=$(ARCH) -O3 -I../../common main-pinned.cu -o run-pinned.o 
	nvcc -arch=$(ARCH) -O3 -I../../common main-managed.cu -o run-managed.o 

clean:
	rm *.o

