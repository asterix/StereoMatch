///////////////////////////////////////////////////////////////////////////
//
// NAME
//  main.cpp -- entry point for command-line stereo matching code
//
// USAGE
//  StereoMatch [option value]*
//
// DESCRIPTION
//  The main routine call the CStereoIO class to process the command-line
//  arguments (pairs of parameters / commands and their values).
//
// SEE ALSO
//  StereoIO.h              description main command-line options
//  StereoParameters.h      description of stereo algorithm parameters
//
// Copyright © Richard Szeliski and Daniel Scharstein, 2001.
// See Copyright.h for more details
//
///////////////////////////////////////////////////////////////////////////

#include "Error.h"
#include "StereoIO.h"
#include "CudaUtilities.h"

extern Timer* profilingTimer;
extern Timer* profilingTimer2;

int main(int argc, const char *argv[])
{
    CStereoIO s;

    // Initialization to work with CUDA capable GPU
    prepareDevice();

    profilingTimer = new Timer;
    profilingTimer2 = new Timer;
    

    try
    {
        s.InterpretCommandLine(argc, argv);
    }
    catch (CError &err)
    {
        fprintf(stderr, err.message);
        fprintf(stderr, "\n");
        return -1;
    }

    // Clear allocated memory
    delete profilingTimer;
    delete profilingTimer2;

#ifdef WIN32 // keep terminal open
    fprintf(stderr, "\nPress any key to continue... ");
    while (1) {
        int c = getchar();
        if (c != EOF) break;
    }
#endif

    return 0;
}
