# Diretórios
SRC_DIR := src
BUILD_DIR := build
METALCPP_DIR := external/metal-cpp  # Ajuste se os headers estiverem em outro lugar

# Arquivos
CPP_SRC := $(SRC_DIR)/main-2.cpp
METAL_SRC := $(SRC_DIR)/MajorityRule-2.metal
METAL_AIR := $(BUILD_DIR)/kernel.air
METAL_LIB := $(BUILD_DIR)/default.metallib
EXEC := $(BUILD_DIR)/main

# Compiladores
CXX := /usr/bin/clang++
METALC := xcrun -sdk macosx metal
METALLIB := xcrun -sdk macosx metallib

# Caminhos para libomp (Apple Silicon)
LIBOMP_PATH := /opt/homebrew/opt/libomp

# Flags
CXXFLAGS := -std=c++17 -stdlib=libc++ -O2 \
	-I$(METALCPP_DIR) \
	-I$(LIBOMP_PATH)/include \
	-L$(LIBOMP_PATH)/lib \
	-Xpreprocessor -fopenmp -lomp \
	-fno-objc-arc \
	-framework Metal -framework Foundation -framework QuartzCore

# Alvo padrão: compila e executa
all: run

# Compilação do shader Metal
$(METAL_LIB): $(METAL_SRC)
	@mkdir -p $(BUILD_DIR)
	$(METALC) -c $< -o $(METAL_AIR)
	$(METALLIB) $(METAL_AIR) -o $(METAL_LIB)

# Compilação do C++
$(EXEC): $(CPP_SRC) $(METAL_LIB)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPP_SRC) -o $(EXEC) $(CXXFLAGS)

# Executa o programa
run: $(EXEC)
	$(EXEC)

# Limpa tudo
clean:
	rm -rf $(BUILD_DIR)
