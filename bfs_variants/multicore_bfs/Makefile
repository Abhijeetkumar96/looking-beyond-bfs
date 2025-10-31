# *__Makefile__*
# target: dependency
# 	action
GXX := g++
CPPFLAGS := -std=c++17 -fopenmp -O3 -Iinclude

OBJDIR := obj
SRCDIR := src
INCDIR := include
BINDIR := bin

# Create the obj directory if it doesn't exist
$(shell mkdir -p $(OBJDIR))
$(shell mkdir -p $(BINDIR))

all: bfs_test
bfs_test: $(OBJDIR)/main.o $(OBJDIR)/multicore_bfs.o
	$(GXX) $(CPPFLAGS) $(OBJDIR)/multicore_bfs.o $(OBJDIR)/main.o -o $(BINDIR)/bfs_test

$(OBJDIR)/multicore_bfs.o: $(SRCDIR)/multicore_bfs.cpp $(INCDIR)/multicore_bfs.hpp $(INCDIR)/thread.hpp
	$(GXX) $(CPPFLAGS) -c $(SRCDIR)/multicore_bfs.cpp -o $(OBJDIR)/multicore_bfs.o

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp $(OBJDIR)/multicore_bfs.o
	$(GXX) $(CPPFLAGS) -c $(SRCDIR)/main.cpp -o $(OBJDIR)/main.o

debug: CPPFLAGS += -DDEBUG
debug: all

clean:
	rm -rf $(OBJDIR)/*.o $(BINDIR)/*

help:
	@echo "Available targets:"
	@echo "  all      : Compiles the main program."
	@echo "  bfs_test : Compiles and links the bfs_test executable."
	@echo "  debug    : Compiles the program with debug symbols."
	@echo "  clean    : Removes object files and executables."

.PHONY: all debug clean help