# CUDA Compiler
NVCC := nvcc
NVCCFLAGS := -O3 -arch=sm_80

# Files
SRCS := bfs.cu expand.cu Graph_CSR.cpp
OBJS := $(SRCS:.cu=.o)
EXEC := bfs_executable

# Target to compile all
all: $(EXEC)

# Rule to make executable
$(EXEC): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Rule to compile .cu files into .o files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(EXEC)
