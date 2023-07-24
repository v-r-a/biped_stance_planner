# This Makefile assumes that you have GLFW libraries and headers installed on,
# which is commonly available through your distro's package manager.
# On Debian and Ubuntu, GLFW can be installed via `apt install libglfw3-dev`.

COMMON=-O2 -I../include -L../lib -std=c++17 -pthread -Wl,-no-as-needed -Wl,-rpath,'$$ORIGIN'/../lib

.PHONY: all
all:
	$(CXX) $(COMMON) bioloid_IK_util.cc   -lmujoco -lglfw   -o ../bin/bioloid_IK_util