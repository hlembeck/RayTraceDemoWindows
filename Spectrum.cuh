#pragma once
#include "ColorModel.cuh"

double* zeroSpectrum();
double* constantSpectrum(double power);
double* bandSpectrum(double power, unsigned int wavelength);