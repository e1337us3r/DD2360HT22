ARCH=sm_86

build:
	nvcc -arch=$(ARCH) --default-stream per-thread -I../../common main.cu -o run.o

plot:
	python3 plot-stream-vs-default.py && python3 plot-stream-count.py

clean:
	rm *.o
