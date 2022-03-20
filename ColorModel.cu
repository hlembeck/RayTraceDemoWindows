#include "ColorModel.cuh"

__device__ double piecewiseGaussian(double x, double m, double s1, double s2) {
	if (x < m) {
		return exp(-0.5 * (x - (double)m) * (x - (double)m) / (s1 * s1));
	}
	return exp(-0.5 * (x - (double)m) * (x - (double)m) / (s2 * s2));
}

__global__ void getColorMatchXYZ(double* output) {
	double l = double(MIN_WAVELENGTH + threadIdx.x * (MAX_WAVELENGTH - MIN_WAVELENGTH) / STEPS);
	switch (blockIdx.x) {
	case 0:
		output[threadIdx.x] = 1.056 * piecewiseGaussian(l, 599.8, 37.9, 31.0) + 0.362 * piecewiseGaussian(l, 442.0, 16.0, 26.7);
		break;
	case 1:
		output[STEPS + threadIdx.x] = 0.821 * piecewiseGaussian(l, 568.8, 46.9, 40.5) + 0.286 * piecewiseGaussian(l, 530.9, 16.3, 31.1);
		break;
	case 2:
		output[2 * STEPS + threadIdx.x] = 1.217 * piecewiseGaussian(l, 437.0, 11.8, 36.0) + 0.681 * piecewiseGaussian(l, 459.0, 26.0, 13.8);
	}
}

__global__ void getTristimulus(double* pixelSpectrums, double* colorMatch, double* output, unsigned int width) {
	double dx = double((MAX_WAVELENGTH - MIN_WAVELENGTH) / STEPS);
	unsigned int index = blockIdx.x * width + blockIdx.y;
	output[3 * index + threadIdx.x] = 0.0;
	for (unsigned int i = 0; i < STEPS; i++) {
		output[3 * index + threadIdx.x] += pixelSpectrums[index * STEPS + i] * colorMatch[threadIdx.x * STEPS + i];
	}
}

__global__ void getRGB(double* xyzPixels, unsigned char* rgbPixels, unsigned int width) {
	unsigned int index = blockIdx.x * width + blockIdx.y;
	switch (threadIdx.x) {
	case 0:
		//Red
		rgbPixels[4 * index] = (unsigned char)min(255.0, 255.0 * (0.41847 * xyzPixels[3 * index] - 0.15866 * xyzPixels[3 * index + 1] - 0.082835 * xyzPixels[3 * index + 2]));
		break;
	case 1:
		//Green
		rgbPixels[4 * index + 1] = (unsigned char)min(255.0, 255.0 * (-0.091169 * xyzPixels[3 * index] + 0.25243 * xyzPixels[3 * index + 1] + 0.015708 * xyzPixels[3 * index + 2]));
		break;
	case 2:
		//Blue
		rgbPixels[4 * index + 2] = (unsigned char)min(255.0, 255.0 * (0.00092090 * xyzPixels[3 * index] - 0.0025498 * xyzPixels[3 * index + 1] + 0.17860 * xyzPixels[3 * index + 2]));
		//Alpha, 255 for PNG, 0 for DIB (convention)
		rgbPixels[4 * index + 3] = 255;
	}
}

__global__ void printTristimulus(double* tristimulusSamples, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		printf("%f %f %f\n", tristimulusSamples[3 * i], tristimulusSamples[3 * i + 1], tristimulusSamples[3 * i + 2]);
	}
	printf("\n");
}

__global__ void printColorMatchXYZ(double* colorMatchXYZ) {
	for (unsigned int i = 0; i < 3; i++) {
		for (unsigned int j = 0; j < STEPS; j++) {
			printf("%f\n", colorMatchXYZ[STEPS * i + j]);
		}
		printf("\n");
	}
}

__global__ void printRGBa(unsigned char* rgba, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		printf("R[%d] G[%d] B[%d] A[%d]\n", rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], rgba[i * 4 + 3]);
	}
}