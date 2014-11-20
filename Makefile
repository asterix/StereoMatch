CPPFLAGS = -O2 -Wall

SRC = BoxFilter.cpp Convert.cpp Convolve.cpp Histogram1D.cpp Image.cpp ImageIO.cpp main.cpp MinFilter.cpp ParameterIO.cpp RefCntMem.cpp StcAggregate.cpp StcDiffusion.cpp StcEvaluate.cpp StcGraphCut.cpp StcOptDP.cpp StcOptimize.cpp StcOptSO.cpp StcPreProcess.cpp StcRawCosts.cpp StcRefine.cpp StcSimulAnn.cpp StereoIO.cpp StereoMatcher.cpp StereoParameters.cpp Warp1D.cpp

HDR = BoxFilter.h Convert.h Convolve.h Copyright.h Error.h Histogram1D.h Image.h ImageIO.h MinFilter.h ParameterIO.h RefCntMem.h StereoIO.h StereoMatcher.h StereoParameters.h Verbose.h Warp1D.h

OBJ = $(SRC:.cpp=.o) maxflow/maxflow.o
 
all: StereoMatch

StereoMatch: $(OBJ)
	g++ -o StereoMatch $(OBJ) -lm

clean:
	rm -f $(OBJ)
