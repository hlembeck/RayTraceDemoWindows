#include "Spectrum.cuh"

double* zeroSpectrum() {
	double* s;
	cudaMalloc((void**)&s, sizeof(double) * STEPS);
	cudaMemset(s, 0, sizeof(double) * STEPS);
	return s;
}

__global__ void setConstantSpectrum(double* spectrum, double power) {
	spectrum[threadIdx.x] = power;
}

double* constantSpectrum(double power) {
	double* s;
	cudaMalloc((void**)&s, sizeof(double) * STEPS);
	setConstantSpectrum << <1, STEPS >> > (s, power);
	return s;
}

__global__ void setBandSpectrum(double* spectrum, double power, unsigned int wavelength) {
	unsigned int stepLength = (MAX_WAVELENGTH - MIN_WAVELENGTH) / STEPS;
	unsigned int index = (wavelength - MIN_WAVELENGTH) / stepLength;
	spectrum[threadIdx.x] = (index == threadIdx.x ? power : 0.0);
}

double* bandSpectrum(double power, unsigned int wavelength) {
	double* s;
	cudaMalloc((void**)&s, sizeof(double) * STEPS);
	setBandSpectrum << <1, STEPS >> > (s, power, wavelength);
	return s;
}