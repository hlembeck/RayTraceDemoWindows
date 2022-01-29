#include "BasicTypes.h"
#include "Spectrum.cuh"

/*
Print Functions
IMPORTANT: Uses printf, must be changed for windows application (alternatively, add a console to the application).
*/
__global__ void printMeshes(Mesh* meshes, unsigned int len);

__global__ void standardPlate(Mesh* mesh, double* spectrum);