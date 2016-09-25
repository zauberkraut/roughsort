name = roughsort
root = $(shell pwd -P)
src = $(root)/src
build = $(root)/build
bin = $(root)/$(name)
testbin = $(build)/test
tex = $(root)/tex

cc = g++
nvcc = nvcc

main = $(build)/main.o
modules = $(build)/random.o $(build)/sort.o $(build)/sort.obj \
          $(build)/util.o $(build)/util.obj
tests = $(build)/test.o

oflags = -O3
wflags = -Wall -Werror
cflags = -std=c++11 -mrdrnd $(oflags) $(wflags) -I/usr/local/cuda/include
nvflags = -std c++11 $(oflags) -Xcompiler "$(wflags)"
libs = -lm -lpthread -L/usr/local/cuda/lib64 -lcudart

$(bin): $(main) $(modules)
	$(cc) -o $(@) $(^) $(libs)

$(build)/%.o: $(src)/%.cxx
	@mkdir -p $(build)
	$(cc) -o $(@) $(cflags) -c $(<)

$(build)/%.obj: $(src)/%.cu
	@mkdir -p $(build)
	$(nvcc) -o $(@) $(nvflags) -c $(<)

debug: oflags = -O0 -g
debug: $(bin)

$(testbin): $(tests) $(modules)
	$(cc) -o $(@) $(^) -lcmocka $(libs)

.PHONY: test clean
test: $(testbin)
	$(^)

clean:
	rm -fr $(build) $(bin) $(tex)/*.aux $(tex)/*.bbl $(tex)/*.blg $(tex)/*.log \
         $(tex)/*.out $(tex)/*.nav $(tex)/*.snm $(tex)/*.toc
