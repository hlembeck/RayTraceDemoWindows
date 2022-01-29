#include "ColorModel.cuh"

double* zeroSpectrum();
double* constantSpectrum(double power);
double* bandSpectrum(double power, unsigned int wavelength);

class SpectrumManager {
public:
	SpectrumManager() : _spectrums({ zeroSpectrum() }) {};
	SpectrumManager(double* s) : _spectrums({ zeroSpectrum(), s }) {};
	~SpectrumManager() { for (double* i : _spectrums) { cudaFree(i); } };

	void push(double* s) { _spectrums.push_back(s); };
	double* getLast() { return _spectrums.back(); };
	double* getZero() { return _spectrums[0]; };
private:
	std::vector<double*> _spectrums;
};