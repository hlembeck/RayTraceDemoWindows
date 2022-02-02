#include "RayTraceEntry.cuh"

bool populateRays(ImgParamPinhole& params, Ray* &rays, cudaError_t& cudaStatus);
bool populateIntersectionData(ImgParamPinhole& params, Ray* &rays, Scene& scene, IntersectionData* &intersectionData, cudaError_t& cudaStatus);
bool populateSamples(ImgParamPinhole& params, IntersectionData* &intersectionData, double* &samples, cudaError_t& cudaStatus);
bool populatePixelSamples(ImgParamPinhole& params, double* &samples, double* &pixelSamples, cudaError_t& cudaStatus);

bool populateRGBQuadArray(ImgParamPinhole& params, unsigned char* &rgbQuadArr, double* pixelSamples, cudaError_t& cudaStatus);

bool getPinholeImage(ImgParamPinhole& params, Scene& scene, unsigned char* &rgbQuadArr) {
	/*
	To store the ray information for each sample ray.
	LOCATION: Device
	LENGTH: width * height * nRays * nRays
	*/
	Ray* rays = 0;
	/*
	To store the spectrumand angle corresponding to the surface hit on the all reflections of all sample rays.
	LOCATION: Device
	LENGTH: params.width * params.height * nRays * nRays * nReflections
	*/
	IntersectionData* intersectionData = 0;
	/*
	To store the raw spectral data computed from intersectionData.
	LOCATION: Device
	LENGTH: width * height * nRays * nRays
	*/
	double* samples = 0;
	/*
	To store the spectra data for each pixel, as computed from samples.
	LOCATION: Device
	LENGTH: width * height
	*/
	double* pixelSamples = 0;

	cudaError_t cudaStatus;

	cudaSetDevice(0);

	//Populate rays with device-allocated data
	if (!populateRays(params, rays, cudaStatus))
		goto error;

	//Populate the intersection data
	if (!populateIntersectionData(params, rays, scene, intersectionData, cudaStatus))
		goto error;

	//At this point, scene can be deallocated
	cudaFree(rays);

	//Populate samples with data from intersectionData
	if (!populateSamples(params, intersectionData, samples, cudaStatus))
		goto error;

	cudaFree(intersectionData);

	//Populate pixel samples using data from samples
	if (!populatePixelSamples(params, samples, pixelSamples, cudaStatus))
		goto error;

	cudaFree(samples);

	//populate rgbBMP with RGB data from pixelSamples. pixelSamples is freed in this function. rgbBMP is allocated in this function, and must be freed from the heap after painting in window.
	if (!populateRGBQuadArray(params, rgbQuadArr, pixelSamples, cudaStatus))
		goto error;

	cudaDeviceReset();

	return true;
error:
	cudaFree(rays);
	cudaFree(intersectionData);
	cudaFree(samples);
	cudaFree(pixelSamples);
	cudaDeviceReset();
	return false;
}

bool populateRays(ImgParamPinhole& params, Ray* &rays, cudaError_t& cudaStatus) {
	cudaStatus = cudaMalloc((void**)&rays, sizeof(Ray) * params.width * params.height * params.nRays * params.nRays);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRays()) Failed to allocate rays.\n"));
		return false;
	}

	dim3 numBlocks(params.height, params.width);
	dim3 numThreadsPerBlock(params.nRays, params.nRays);

	generateRaysPinhole << <numBlocks, numThreadsPerBlock >> > (rays, params.pinhole, params.sensorHeight / 2.0, -params.sensorWidth / 2.0, params.sensorWidth / (double)params.width, params.sensorWidth / ((double)params.width * (double)params.nRays), params.width, params.height, params.nRays);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRays()) Failed generateRaysPinhole().\n"));
		return false;
	}
	return true;
}

bool populateIntersectionData(ImgParamPinhole& params, Ray* &rays, Scene& scene, IntersectionData* &intersectionData, cudaError_t& cudaStatus) {
	cudaStatus = cudaMalloc((void**)&intersectionData, sizeof(IntersectionData) * params.width * params.height * params.nRays * params.nRays * params.nReflections);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateIntersectionData()) Failed malloc of intersectionData.\n"));
		return false;
	}

	dim3 numBlocks(params.height, params.width);
	dim3 numThreadsPerBlock(params.nRays, params.nRays);
	for (unsigned int i = 0; i < params.nReflections; i++) {
		traceRays << <numBlocks, numThreadsPerBlock >> > (rays, scene.meshes, scene.len, intersectionData + i * sizeof(IntersectionData), params.width, params.nRays, params.nReflections);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			TCHAR buf[256] = L"";
			memset(buf, 0, sizeof(TCHAR) * 256);
			wsprintf(buf, L"(In populateIntersectionData()) Failed traceRays() on iteration %d.\n", i);
			OutputDebugString(buf);
			return false;
		}
	}
	return true;
}

bool populateSamples(ImgParamPinhole& params, IntersectionData* &intersectionData, double* &samples, cudaError_t& cudaStatus) {
	cudaStatus = cudaMalloc((void**)&samples, sizeof(double) * params.width * params.height * params.nRays * params.nRays * STEPS);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateSamples()) Failed malloc of samples.\n"));
		return false;
	}

	dim3 numBlocks(params.height, params.width);
	dim3 numThreadsPerBlock(params.nRays, params.nRays);

	computeSamples << <numBlocks, numThreadsPerBlock >> > (intersectionData, samples, params.width, params.nRays, params.nReflections);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateSamples()) Failed computeSamples.\n"));
		return false;
	}
	return true;
}

bool populatePixelSamples(ImgParamPinhole& params, double* &samples, double* &pixelSamples, cudaError_t& cudaStatus) {
	cudaStatus = cudaMalloc((void**)&pixelSamples, sizeof(double) * params.width * params.height * STEPS);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populatePixelSamples()) Failed to allocate pixelSamples.\n"));
		return false;
	}
	dim3 numBlocks(params.height, params.width);
	computePixels << <numBlocks, 1 >> > (samples, pixelSamples, params.width, params.nRays);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populatePixelSamples()) Failed computePixels().\n"));
		return false;
	}
}

bool populateRGBQuadArray(ImgParamPinhole& params, unsigned char* &rgbQuadArr, double* pixelSamples, cudaError_t& cudaStatus) {

	double* colorMatchXYZ = 0;
	double* tristimulusPixels = 0;
	unsigned char* rgbPixels = 0;

	cudaStatus = cudaMalloc((void**)&colorMatchXYZ, sizeof(double) * 3 * STEPS);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed malloc of colorMatchXYZ.\n"));
		return false;
	}

	getColorMatchXYZ << <3, STEPS >> > (colorMatchXYZ);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed getColorMatchXYZ().\n"));
		cudaFree(colorMatchXYZ);
		cudaFree(pixelSamples);
		return false;
	}

	cudaStatus = cudaMalloc((void**)&tristimulusPixels, sizeof(double) * 3 * params.width * params.height);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed malloc of tristimulusPixels.\n"));
		cudaFree(colorMatchXYZ);
		cudaFree(pixelSamples);
		return false;
	}

	dim3 numBlocks(params.height, params.width);

	getTristimulus << <numBlocks, 3 >> > (pixelSamples, colorMatchXYZ, tristimulusPixels, params.width);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed getTristimulus().\n"));
		cudaFree(colorMatchXYZ);
		cudaFree(tristimulusPixels);
		cudaFree(pixelSamples);
		return false;
	}

	cudaStatus = cudaFree(pixelSamples);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed free of pixelSamples.\n"));
		return false;
	}

	cudaStatus = cudaFree(colorMatchXYZ);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed free of colorMatchXYZ.\n"));
		return false;
	}

	cudaStatus = cudaMalloc((void**)&rgbPixels, 4 * params.width * params.height);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed malloc of rgbPixels.\n"));
		cudaFree(colorMatchXYZ);
		cudaFree(tristimulusPixels);
		cudaFree(pixelSamples);
		return false;
	}

	getRGB << <numBlocks, 3 >> > (tristimulusPixels, rgbPixels, params.width);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed getRGB.\n"));
		cudaFree(colorMatchXYZ);
		cudaFree(tristimulusPixels);
		cudaFree(pixelSamples);
		cudaFree(rgbPixels);
		return false;
	}

	cudaStatus = cudaFree(tristimulusPixels);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed free of tristimulusPixels.\n"));
		return false;
	}

	rgbQuadArr = new unsigned char[4 * params.width * params.height];

	cudaStatus = cudaMemcpy(rgbQuadArr, rgbPixels, 4 * params.width * params.height, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed memcopy of rgbPixels -> rgbBMP.\n"));
		cudaFree(colorMatchXYZ);
		cudaFree(tristimulusPixels);
		cudaFree(pixelSamples);
		cudaFree(rgbPixels);
		return false;
	}

	cudaStatus = cudaFree(rgbPixels);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateRGBQuadArray()) Failed free of rgbPixels.\n"));
		return false;
	}

	return true;
}