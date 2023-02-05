PROGRAM=encode_stb
NVCC=nvcc
OPT= -g -G
all: encode_stb

main: encode_stb.o encode.o
	${NVCC} ${OPT} -o main encode_stb.o

encode_stb.o: encode_stb.cu
	$(NVCC) ${OPT} $(CUDAFLAGS)	-std=c++11 -c encode_stb.cu

encode.o: encode.cu
	$(NVCC) ${OPT} $(CUDAFLAGS)	-std=c++11 -c encode.cu

encode_stb: encode_stb.o encode.o
	${NVCC} ${CUDAFLAGS} -o encode_stb encode.o encode_stb.o
	
clean:
	${RM} *.o encode_stb

