all:
	nvcc -ccbin g++ -I/usr/local/cuda/include  -m64   -gencode arch=compute_30,code=compute_30 -o sum_kernel.ptx -ptx sum_kernel.cu
	g++ -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -fPIC -o code.o -c code.cpp
	swig -c++ -python code.i
	g++ -c -fPIC code_wrap.cxx -I/usr/include/python3.5 -L/usr/lib/python3.5
	g++ -shared -Wl,-soname,_code.so -o _code.so code.o code_wrap.o -L/usr/local/cuda/lib64/stubs -lcuda
