#include "BasicTypes.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//BLUE WAVELENGTH ~450NM
//GREEN WAVELENGTH ~545NM
/*
getColorMatchXYZ populates the array output[3*STEPS] with the values of the XYZ color matching functions x,y,z (see https://en.wikipedia.org/wiki/CIE_1931_color_space). The mappings are interpreted as follows (l is a wavelength (nm) in the range [MIN_WAVELENGTH,MAX_WAVELENGTH]):
x(l) = output[l - MIN_WAVELENGTH],
y(l) = output[STEPS + l - MIN_WAVELENGTH],
z(l) = output[2*STEPS + l - MIN_WAVELENGTH]
*/
__global__ void getColorMatchXYZ(double* output);

__global__ void getTristimulus(double* pixelSpectrums, double* colorMatch, double* output, unsigned int width);

__global__ void getRGB(double* xyzPixels, unsigned char* rgbPixels, unsigned int width);

__global__ void printTristimulus(double* tristimulusSamples, unsigned int len);

__global__ void printColorMatchXYZ(double* colorMatchXYZ);

__global__ void printRGBa(unsigned char* rgba, unsigned int len);