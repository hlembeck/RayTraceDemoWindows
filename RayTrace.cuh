#include "Spectrum.cuh"

/*
Print Functions
IMPORTANT: Uses printf, must be changed for windows application (alternatively, add a console to the application).
*/
__global__ void printSamples(double* samples, unsigned int len);
__global__ void printRays(Ray* rays, unsigned int len);

__global__ void generateRaysPinhole(Ray* rays, Triple<double> pinhole, double top, double left, double pixelSize, double raySpace, unsigned int width, unsigned int height, unsigned int nRays);

__global__ void traceRays(Ray* rays, Face* faces, unsigned int numFaces, IntersectionData* data, unsigned int width, unsigned int nRays, unsigned int nReflections);

__global__ void computeSamples(IntersectionData* data, double* samples, unsigned int width, unsigned int nRays, unsigned int nReflections);

__global__ void computePixels(double* samples, double* pixels, unsigned int width, unsigned int nRays);

__global__ void setSpectrums(Face* faces, unsigned int len, double* spectrums);