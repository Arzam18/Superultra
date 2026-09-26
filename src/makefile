CXXFLAGS_COMMON = -std=c++17 -O3 -funroll-loops -ffast-math -Wall -Wextra -flto

ifeq ($(ARCH),android-neon)
    CXXFLAGS = $(CXXFLAGS_COMMON) \
               -march=armv8-a+simd \
               -mtune=cortex-a76 \
               -MMD -MP

    LDFLAGS = -flto -static-libstdc++ -static-libgcc

    TARGET = Superultra-2.1
else
    CXXFLAGS = $(CXXFLAGS_COMMON) \
               -march=native \
               -MMD -MP

    LDFLAGS = -flto

    TARGET = Superultra-2.1
endif

SOURCES = attacks.cpp \
          eval.cpp \
          main.cpp \
          movegen.cpp \
          movemake.cpp \
          movescore.cpp \
          nnue.cpp \
          search.cpp \
          see.cpp \
          timecontrol.cpp \
          tt.cpp \
          uci.cpp

OBJECTS = $(SOURCES:.cpp=.o)
DEPS = $(OBJECTS:.o=.d)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

-include $(DEPS)

clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGET)
