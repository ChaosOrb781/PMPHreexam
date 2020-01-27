CXX        = g++
CXXFLAGS   = -fopenmp -O3 -DWITH_FLOATS=0 -std=c++11 -lcuda -lcudart
LIB        = -L$(OPENCL_LIBDIR) -lOpenCL

INCLUDES  += -I ../include

SOURCES_CPP =ProjectMain.cpp
HELPERS     =../include/Constants.h ../include/ParseInput.h ../include/ParserC.h ../include/OpenmpUtil.h
OBJECTS     =ProjectMain.o
EXECUTABLE  =runproject


default: cpu

.cpp.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

cpu: $(EXECUTABLE)
$(EXECUTABLE): kernel.o $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXECUTABLE) $(OBJECTS) kernel.o

kernel.o:
	nvcc -c -arch=sm_20 CUDAKernels.cu


run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)

