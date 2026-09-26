# Variables
CXX = g++
CXXFLAGS = -std=c++17 -O3 -funroll-loops -ffast-math -ftree-vectorize -ftree-loop-vectorize -Wall -Wextra -flto
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
DEP = $(SRC:.cpp=.d)

# Example: "make ARCH=skylake-avx512"
ARCH ?= native

ifeq ($(ARCH),android-neon)
	ARCHFLAGS = -march=armv8-a+simd -mtune=cortex-a76
else
	ARCHFLAGS = -march=$(ARCH) -mtune=$(ARCH)
endif

# Android uses the NDK C++ runtime and does not support the desktop -static
# link used by the original Makefile.
ifeq ($(ANDROID),1)
	LDFLAGS = -static-libstdc++ -Wl,-z,stack-size=8388608
	NAME = Superultra-2.1-android-arm64
	RM = rm -f
else ifeq ($(OS), Windows_NT)
	LDFLAGS = -Wl,--stack=8388608,--no-whole-archive -static
	NAME = Superultra-2.1.exe
	RM = del -f
else
	LDFLAGS = -Wl,--no-whole-archive -static
	NAME = Superultra-2.1
	RM = rm -f
endif

# All .o ---> executable
$(NAME): $(OBJ)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(LDFLAGS) -pthread -o $@ $^

# .cpp ---> .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) -MMD -MP -c -o $@ $<

.PHONY: clean
clean:
	$(RM) $(OBJ) $(DEP) $(NAME)

-include $(DEP)
