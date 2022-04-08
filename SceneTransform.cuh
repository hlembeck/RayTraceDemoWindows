#pragma once
#include "Spectrum.cuh"

//Functions returning transform matrices allocated on device.
double* getTranslateTransform(Triple<double>& p, cudaError_t& cudaStatus);
double* getScaleTransform(Triple<double>& v, cudaError_t& cudaStatus);
double* getRotateTransform(Triple<double>& v, double a, cudaError_t& cudaStatus);
double* getRotateTransformHOST(Triple<double>& v, double a);

//Kernel for scene transform with respect to a transformation matrix.
__global__ void transformMeshes(Mesh* meshes, double* mat);

void transformMeshHOST(Face* faces, unsigned int len, double* mat);

//Store a=a*b
void multiply(double* a, double*& b);