name = roughsort
root = $(shell pwd -P)
src = $(root)/src
build = $(root)/build
bin = $(root)/$(name)
testbin = $(build)/test
tex = $(root)/tex

cc = nvcc

main = $(build)/main.o
modules = $(build)/random.o $(build)/sort.o $(build)/sort.obj \
          $(build)/util.o $(build)/util.obj
tests = $(build)/test.o

oflags = -O4
wflags = -Wall -Werror
# note GCC RDRAND flag "-mrdrnd"
cflags = -std c++11 $(oflags) -Xcompiler "-mrdrnd $(wflags)"

$(bin): $(main) $(modules)
	$(cc) -o $(@) $(^)

$(build)/%.o: $(src)/%.cxx
	@mkdir -p $(build)
	$(cc) -o $(@) $(cflags) -c $(<)

$(build)/%.obj: $(src)/%.cu
	@mkdir -p $(build)
	$(cc) -o $(@) $(cflags) -c $(<)

debug: oflags = -O0 -g
debug: $(bin)

$(testbin): $(tests) $(modules)
	$(cc) -o $(@) $(^) -lcmocka

.PHONY: test clean
test: $(testbin)
	$(^)

clean:
	rm -fr $(build) $(bin) $(tex)/*.aux $(tex)/*.bbl $(tex)/*.blg $(tex)/*.log \
         $(tex)/*.out $(tex)/*.nav $(tex)/*.snm $(tex)/*.toc
