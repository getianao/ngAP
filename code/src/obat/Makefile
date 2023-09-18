all: nvcc clang

nvcc:
	nvcc -O0 --std=c++11 -Xptxas='-v' --source-in-ptx -m64 main.cu one_byte_at_a_time.cu -arch=sm_50 -I../../include -L../../build/lib -lgpunfacommons -lgpunfautils -keep -o obat1_nvcc

clang:
	clang++ -O0 --std=c++11 main.cu one_byte_at_a_time.cu --cuda-path=${CUDA_ROOT} --cuda-gpu-arch=sm_50 -lcudart_static -ldl -lrt -pthread -L/home/hyliu/gcc65/install/lib64  -lstdc++ -D_GLIBCXX_USE_CXX11_ABI=0 -L${CUDA_ROOT}/lib64 -I../../include -L../../build/lib -lgpunfacommons -lgpunfautils -save-temps -o obat1_clang

clean:
	rm -f *.o
	rm -f *.ii
	rm -f *.i
	rm -f *.ptx
	rm -f *fatbin*
	rm -f *cubin*
	rm -f *stub*
	rm -f *sm_*
	rm -f *cudafe*
	rm -f *module_id
	rm -f *dlink*
	rm -f a.out
	rm -f *.png
	rm -f *.txt
	rm -f *.ll
	rm -f *.bc
	rm -f *.s
	rm -f *.cui
	rm -f obat1*
