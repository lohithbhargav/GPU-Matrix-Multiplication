all: app0 app1 app2 app3 app4 

app0:
	g++ matrixMul_cpu.cpp -o matrixMul_cpu.exe

app1:
	nvcc matrixMul_gpu.cu -o matrixMul_gpu.exe

app2:
	nvcc matrixMul_gpu_v2.cu -o matrixMul_gpu_v2.exe

app3:
	nvcc matrixMul_gpu_v3.cu -o matrixMul_gpu_v3.exe

app4:
	nvcc matrixMul_gpu_v4.cu -o matrixMul_gpu_v4.exe

clean:
	rm -rf *.exe
