ARCH=sm_86

build:
	nvcc -arch=$(ARCH) -O3 -I../../common -lcusparse -lcublas main.cu -o run.o 

plot:
	python3 plot.py

clean:
	rm *.o

