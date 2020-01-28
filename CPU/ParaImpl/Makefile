CXX        = g++
CXXFLAGS   = -fopenmp -O3 -DWITH_FLOATS=0 -std=c++11 -lcuda -lcudart -lcudadevrt
LIB        = -L$(OPENCL_LIBDIR) -lOpenCL

INCLUDES  += -I ../include

SOURCES_CPP =ProjectMain.cpp
HELPERS     =../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/OpenmpUtil.h ../include/TridagKernel.cu.h ../include/CudaUtilProj.cu.h ./KernelAlgorithm.h
OBJECTS     =ProjectMain.o
EXECUTABLE  =runproject


default: cpu

.cpp.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cpu: $(EXECUTABLE)
$(EXECUTABLE): KernelAlgorithm.o $(OBJECTS)
	$(CXX) -o $(EXECUTABLE) $(CXXFLAGS) $(INCLUDES) $(OBJECTS) KernelAlgorithm.o

KernelAlgorithm.o:
	nvcc $(INCLUDES) -arch=compute_75 -rdc=true -dc KernelAlgorithm.cu -o KernelAlgorithm_nolink.o -lcudart -lcudadevrt
	nvcc -arch=compute_75 -dlink -rdc=true -o KernelAlgorithm.o KernelAlgorithm_nolink.o -lcudart -lcudadevrt


run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)

