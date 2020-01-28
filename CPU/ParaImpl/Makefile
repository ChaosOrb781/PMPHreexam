CXX        = g++
CXXFLAGS   = -fopenmp -O3 -DWITH_FLOATS=0 -std=c++11 -lcuda -lcudart
LIB        = -L$(OPENCL_LIBDIR) -lOpenCL

INCLUDES  += -I ../include

HELPERS     =../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/OpenmpUtil.h ../include/TridagKernel.cu.h ../include/CudaUtilProj.cu.h

SOURCES_CPP_1 =ProjectMain1.cpp
OBJECTS_1     =ProjectMain1.o
EXECUTABLE_1  =runproject1


default: cpu

.cpp.o: $(SOURCES_CPP_1) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cpu: $(EXECUTABLE_1)
$(EXECUTABLE_1): KernelAlgorithm.o $(OBJECTS_1)
	$(CXX) -o $(EXECUTABLE_1) $(CXXFLAGS) $(INCLUDES) $(OBJECTS_1) KernelAlgorithm.o

KernelAlgorithm.o:
	nvcc $(INCLUDES) -c KernelAlgorithm.cu


run_small: $(EXECUTABLE_1)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE_1) 2> Debug.txt

run_medium: $(EXECUTABLE_1)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE_1) 2> Debug.txt

run_large: $(EXECUTABLE_1)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE_1) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS_1)

