#include "RayTraceEntry.cuh"

bool populateRays(ImgParamPinhole& params, Ray* &rays, cudaError_t& cudaStatus);
bool populateIntersectionData(ImgParamPinhole& params, Ray* &rays, Face* faces, unsigned int numFaces, IntersectionData* &intersectionData, cudaError_t& cudaStatus);
bool populateSamples(ImgParamPinhole& params, IntersectionData* &intersectionData, double* &samples, cudaError_t& cudaStatus);
bool populatePixelSamples(ImgParamPinhole& params, double* &samples, double* &pixelSamples, cudaError_t& cudaStatus);
bool populateRGBQuadArray(ImgParamPinhole& params, unsigned char* &rgbQuadArr, double* pixelSamples, cudaError_t& cudaStatus);

bool paramsToDevice(SceneParams& sceneParams, double*& spectrums, double*& spectrumsBack, Face*& faces, unsigned int*& meshes, cudaError_t& cudaStatus);


bool getPinholeImage(RTParams& rtParams) {
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
	/*
	To store the spectrums that the meshes reference.
	LOCATION: Values stored on Device
	*/
	double* spectrums = 0;
	double* spectrumsBack = 0;
	/*
	Array of Face structs that represents the faces present in a scene.
	LOCATION: Device.
	*/
	Face* faces = 0;
	/*
	Array of indices of faces that partitions faces into meshes.
	LOCATION: Device
	*/
	unsigned int* meshes = 0;
	//unsigned int numSpectrums = rtParams.sceneParams.spectrums.size();

	cudaError_t cudaStatus;

	cudaSetDevice(0);

	if (!paramsToDevice(rtParams.sceneParams, spectrums, spectrumsBack, faces, meshes, cudaStatus))
		goto error;

	//Populate rays with device-allocated data
	if (!populateRays(rtParams.params, rays, cudaStatus))
		goto error;

	//Populate the intersection data
	if (!populateIntersectionData(rtParams.params, rays, faces, rtParams.sceneParams.faces.size(), intersectionData, cudaStatus))
		goto error;

	cudaStatus = cudaFree(rays);
	cudaStatus = cudaFree(faces);
	cudaStatus = cudaFree(meshes);
	if (cudaStatus != cudaSuccess) {
		printf("Error freeing rays, faces, or meshes.\n");
		goto error;
	}

	//Populate samples with data from intersectionData
	if (!populateSamples(rtParams.params, intersectionData, samples, cudaStatus))
		goto error;

	cudaStatus = cudaFree(spectrums);
	cudaStatus = cudaFree(spectrumsBack);
	cudaStatus = cudaFree(intersectionData);
	if (cudaStatus != cudaSuccess) {
		printf("Failed to free intersectionData or spectrums.\n");
		goto error;
	}

	//Populate pixel samples using data from samples
	if (!populatePixelSamples(rtParams.params, samples, pixelSamples, cudaStatus))
		goto error;

	cudaStatus = cudaFree(samples);
	if (cudaStatus != cudaSuccess) {
		printf("Failed to free samples.\n");
		goto error;
	}

	//populate rgbBMP with RGB data from pixelSamples. pixelSamples is freed in this function. rgbBMP is allocated in this function, and must be freed from the heap after painting in window.
	if (!populateRGBQuadArray(rtParams.params, rtParams.rgbQuadArr, pixelSamples, cudaStatus))
		goto error;

	cudaDeviceReset();

	return true;
error:
	OutputDebugString(TEXT("(In getPinholeImage()) There was an error.\n"));
	cudaFree(rays);
	cudaFree(intersectionData);
	cudaFree(spectrums);
	cudaFree(spectrumsBack);
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

bool populateIntersectionData(ImgParamPinhole& params, Ray* &rays, Face* faces, unsigned int numFaces, IntersectionData* &intersectionData, cudaError_t& cudaStatus) {
	unsigned int planeLength = params.width * params.height * params.nRays * params.nRays;
	cudaStatus = cudaMalloc((void**)&intersectionData, sizeof(IntersectionData) * params.width * params.height * params.nRays * params.nRays * (params.nReflections + 1));
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In populateIntersectionData()) Failed malloc of intersectionData.\n"));
		return false;
	}

	dim3 numBlocks(params.height, params.width);
	dim3 numThreadsPerBlock(params.nRays, params.nRays);
	for (unsigned int i = 0; i < params.nReflections + 1; i++) {
		traceRays << <numBlocks, numThreadsPerBlock >> > (rays, faces, numFaces, intersectionData + i, params.width, params.nRays, params.nReflections + 1);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			TCHAR buf[256] = L"";
			memset(buf, 0, sizeof(TCHAR) * 256);
			wsprintf(buf, L"(In populateIntersectionData()) Failed traceRays() on iteration %d.\n", i);
			OutputDebugString(buf);
			return false;
		}
		//printf("\n\n\n");
		/*printRays << <1, 1 >> > (rays, params.height * params.width * params.nRays * params.nRays);
		printf("\n\n\n\n");*/
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

	computeSamples << <numBlocks, numThreadsPerBlock >> > (intersectionData, samples, params.width, params.nRays, params.nReflections + 1);
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

bool paramsToDevice(SceneParams& sceneParams, double*& spectrums, double*& spectrumsBack, Face*& faces, unsigned int*& meshes, cudaError_t& cudaStatus) {
	unsigned int numBlocks;

	cudaStatus = cudaMalloc((void**)&spectrums, sizeof(double) * sceneParams.spectrums.size());
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In paramsToDevice()) Failed cudaMalloc of spectrums.\n"));
		return false;
	}
	cudaStatus = cudaMemcpy(spectrums, sceneParams.spectrums.data(), sizeof(double) * sceneParams.spectrums.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In paramsToDevice()) Failed cudaMemcpy of spectrums.\n"));
		return false;
	}

	cudaStatus = cudaMalloc((void**)&spectrumsBack, sizeof(double) * sceneParams.spectrums.size());
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In paramsToDevice()) Failed cudaMalloc of spectrumsBack.\n"));
		return false;
	}
	cudaStatus = cudaMemcpy(spectrumsBack, sceneParams.spectrumsBack.data(), sizeof(double) * sceneParams.spectrums.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In paramsToDevice()) Failed cudaMemcpy of spectrumsBack.\n"));
		return false;
	}

	cudaStatus = cudaMalloc((void**)&faces, sizeof(Face) * sceneParams.faces.size());
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In paramsToDevice()) Failed cudaMalloc of faces.\n"));
		return false;
	}
	cudaStatus = cudaMemcpy(faces, sceneParams.faces.data(), sizeof(Face) * sceneParams.faces.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In paramsToDevice()) Failed cudaMemcpy of faces.\n"));
		return false;
	}

	numBlocks = (sceneParams.faces.size() & 511 ? (sceneParams.faces.size() >> 9) + 1 : (sceneParams.faces.size() >> 9));
	setSpectrums << <numBlocks, 512 >> > (faces, sceneParams.faces.size(), spectrums, spectrumsBack);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In paramsToDevice()) Failed setSpectrums().\n"));
		return false;
	}

	cudaStatus = cudaMalloc((void**)&meshes, sizeof(unsigned int) * sceneParams.meshes.size());
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In paramsToDevice()) Failed cudaMalloc of meshes.\n"));
		return false;
	}
	cudaStatus = cudaMemcpy(meshes, &sceneParams.meshes[0], sizeof(unsigned int) * sceneParams.meshes.size(), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		OutputDebugString(TEXT("(In paramsToDevice()) Failed cudaMemcpy of meshes.\n"));
		return false;
	}

	return true;
}