.PHONY: directories


#ifeq ($(CXXFLAGS),)
CC = g++
AR = ar
NVCC=/usr/local/cuda/bin/nvcc
SM=sm_80

# -I../mnrl/include
CXXFLAGS= -std=c++14 -O3 -Iinclude -I../../include  -I../../include/commons -I../../include/pugixml -I../../include/gpunfautils -w
CUDA_INCLUDE=-I/usr/local/cuda/include
NVCCFLAGS+= $(CUDA_INCLUDE) -O3 -D_FORCE_INLINES -arch ${SM}  --default-stream per-thread --compiler-options -Wno-deprecated -lineinfo  -cudart shared -rdc=true -use_fast_math -extra-device-vectorization -restrict 
# -Xptxas -v
#endif

#NVCCFLAGS += -I../cub #-I../obat/include

objects := obj/run_ahead_approach.o obj/main.o
libs_objects = obj/run_ahead_approach.o

LDFLAGS= -L../../build/lib -lgpunfacommons -lgpunfautils -lpthread -ltbb


all: directories bin/asyncap

directories:
	mkdir -p obj
	mkdir -p export_lib
	mkdir -p bin

export_lib/libgpunfa_runahead.so: $(libs_objects)
	$(NVCC)  ${NVCCFLAGS} ${CXXFLAGS}  -shared  --compiler-options '-fPIC' ${libs_objects}  ${LDFLAGS} -o $@ 


# bin/asyncap: ${objects} export_lib/libgpunfa_runahead.so
bin/asyncap: ${objects}
	$(NVCC) ${NVCCFLAGS}   ${CXXFLAGS} ${objects} ${LDFLAGS} -o $@ 
	cp -r bin/* ../../build/bin/

obj/run_ahead_approach.o: 
	$(NVCC) -c ${CXXFLAGS} ${NVCCFLAGS} --shared --compiler-options '-fPIC'  src/run_ahead_approach.cu ${LDFLAGS} -o $@

obj/main.o:
	nvcc -c ${CXXFLAGS} ${NVCCFLAGS} src/main.cu ${LDFLAGS} -o $@

clean:
	rm -rf obj
	rm -rf export_lib

