# ==========================================================
# Makefile for BFS Adam (CUDA Implementation)
# ==========================================================

# --- Compiler and Flags ---
NVCC      := nvcc
CUDA_ARCH := sm_89
CXXFLAGS  := -std=c++17 -O3 -arch=$(CUDA_ARCH)

# --- Target and Source ---
TARGET    := bin/bfs_adam
SOURCE    := main.cu

# ==========================================================
# Rules
# ==========================================================

# Default build rule
all: $(TARGET)

# Build target
$(TARGET): $(SOURCE)
	@mkdir -p $(dir $@)
	$(NVCC) $(CXXFLAGS) $< -o $@
	@echo "✅ Build complete: $@"

# Clean rule
clean:
	@echo "🧹 Cleaning up..."
	rm -f $(TARGET)

# Force rebuild
rebuild: clean all

# Phony targets
.PHONY: all clean rebuild

