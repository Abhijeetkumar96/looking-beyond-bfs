# Makefile for building the BFS Adam using CUDA

# Compiler settings
CC=nvcc
CFLAGS=-std=c++17 -O3 -arch=sm_80

# Build target
TARGET=bfs_adam

# Source file
SOURCE=main.cu

# Build rule for the main target
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)

# Clean rule for cleaning up the build
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: clean
