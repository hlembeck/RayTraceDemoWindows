#pragma once
#include "BasicTypes.h"
#include "SceneTransform.cuh"

/*
Print Functions
IMPORTANT: Uses printf, must be changed for windows application (alternatively, add a console to the application).
*/
__global__ void printMeshes(Mesh* meshes, unsigned int len);

__global__ void standardPlate(Mesh* mesh, double* spectrum);

__global__ void standardPrism(Mesh* mesh, double* spectrum);

Mesh standardPlateHOST(double* spectrum);

Mesh standardPrismHOST(double* spectrum);

void addPlateHOST(std::vector<Face>& faceVector, double* transformMatrix, unsigned int spectrum, double reflectivity, double refractiveIndex);

void addPrismHOST(std::vector<Face>& faceVector, double* transformMatrix, unsigned int spectrum, double reflectivity, double refractiveIndex);