CXX ?= g++
CXXFLAGS ?= -O3 -march=native -DNDEBUG -std=c++17 -Wall -Wextra -fopenmp

.PHONY: all clean
all: csc_light_preprocess_bin

csc_light_preprocess_bin: csc_light_preprocess.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f csc_light_preprocess_bin
