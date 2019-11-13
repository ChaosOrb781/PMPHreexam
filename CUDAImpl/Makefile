COMPILER   = nvcc
CFLAGS     = -O3 -c
LDFLAGS    = -lOpenCL 

SOURCES    = ProjectMain.cu ProjCoreOrig.cu ProjHelperFun.cu
OBJECTS    = $(SOURCES:.cu=.o)

HELPERS    = ./includes/ProjHelperFun.cu.h ./includes/Constants.cu.h ./includes/ParseInput.cu.h ./includes/ParserC.cu.h ./includes/OpenmpUtil.cu.h ./includes/CudaUtilProj.cu.h ./kernels/CUDAKernels.cu.h ./kernels/TridagKernel.cu.h
EXECUTABLE = runproject

default: compile run

all: $(EXECUTABLE)

%.o: %.cu
	$(COMPILER) $(CFLAGS) $< -o $@ 

$(EXECUTABLE): $(OBJECTS)
	$(COMPILER) $(LDFLAGS) $(OBJECTS) -o $@

run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f $(EXECUTABLE) $(OBJECTS)
