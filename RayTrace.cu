#include "RayTrace.cuh"

//store v*w in v
__device__ void cross(Triple<double>& v, Triple<double>& w) {
	double a, b;
	a = v.y * w.z - v.z * w.y;
	b = v.z * w.x - v.x * w.z;
	v.z = v.x * w.y - v.y * w.x;
	v.x = a;
	v.y = b;
}

__device__ double dot(Triple<double>& v, Triple<double>& w) {
	return v.x * w.x + v.y * w.y + v.z * w.z;
}

__global__ void printSamples(double* samples, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		for (unsigned int j = 0; j < STEPS; j++) {
			printf(" %f , ", samples[i * STEPS + j]);
		}
		printf("\n\n");
	}
}

__global__ void printRays(Ray* rays, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		printf("Origin: %f %f %f\n", rays[i].o.x, rays[i].o.y, rays[i].o.z);
		printf("Direction: %f %f %f\n\n", rays[i].d.x, rays[i].d.y, rays[i].d.z);
	}
}

__global__ void printFaces(Face* faces, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		printf("(%d) Points:\n", i);
		printf(" %g %g %g\n", faces[i].p1.x, faces[i].p1.y, faces[i].p1.z);
		printf(" %g %g %g\n", faces[i].p2.x, faces[i].p2.y, faces[i].p2.z);
		printf(" %g %g %g\n", faces[i].p3.x, faces[i].p3.y, faces[i].p3.z);
		printf(" Normal: %g %g %g\n\n", faces[i].n.x, faces[i].n.y, faces[i].n.z);
	}
}

__global__ void printIntersectionData(IntersectionData* data, unsigned int len) {
	for (unsigned int i = 0; i < len; i++) {
		printf("Angle: %g , Spectrum: %p\n", data[i].angle, data[i].spectrum);
	}
	printf("\n\n");
}

__global__ void generateRaysPinhole(Ray* rays, Triple<double> pinhole, double top, double left, double pixelSize, double raySpace, unsigned int width, unsigned int height, unsigned int nRays) {
	unsigned int index = (blockIdx.x * width + blockIdx.y) * nRays * nRays + threadIdx.x * nRays + threadIdx.y;
	double x = left + blockIdx.y * pixelSize + (threadIdx.y + 0.5) * raySpace;
	double y = top - blockIdx.x * pixelSize - (threadIdx.x + 0.5) * raySpace;
	rays[index].o.x = x;
	rays[index].o.y = y;
	rays[index].o.z = 0.0;

	rays[index].d.x = pinhole.x - x;
	rays[index].d.y = pinhole.y - y;
	rays[index].d.z = pinhole.z;
	rays[index].index = -1;
}

__device__ double timeOfIntersection(Ray& ray, Face& face) {
	//(o+vt-p).n = 0 , (o-p).n + t(v.n) = 0 , t = (p-o).n/(v.n)
	double t = ray.d.x * face.n.x + ray.d.y * face.n.y + ray.d.z * face.n.z;
	if (t) {
		t = ((face.p1.x - ray.o.x) * face.n.x + (face.p1.y - ray.o.y) * face.n.y + (face.p1.z - ray.o.z) * face.n.z) / t;
		return t;
	}
	return DBL_MAX;
}

__device__ unsigned int smallestTime(double* times, unsigned int len) {
	double min = DBL_MAX;
	unsigned int j = 0;
	for (unsigned int i = 0; i < len; i++) {
		if (times[i] < min) {
			j = i;
			min = times[i];
		}
	}
	return j;
}

__global__ void testFaces(Ray& ray, Face* faces, unsigned int len, double* times) {
	unsigned int index = blockIdx.x * 512 + threadIdx.x;
	if (index < len) {
		if (ray.index == index) {
			//printf("test\n");
			times[index] = DBL_MAX;
			return;
		}
		double time = timeOfIntersection(ray, faces[index]);
		if (time <= 0.0) {
			times[index] = DBL_MAX;
			return;
		}
		Triple<double> a = { ray.o.x + time * ray.d.x - faces[index].p1.x , ray.o.y + time * ray.d.y - faces[index].p1.y , ray.o.z + time * ray.d.z - faces[index].p1.z };
		Triple<double> b = { ray.o.x + time * ray.d.x - faces[index].p2.x , ray.o.y + time * ray.d.y - faces[index].p2.y , ray.o.z + time * ray.d.z - faces[index].p2.z };
		Triple<double> c = { ray.o.x + time * ray.d.x - faces[index].p3.x , ray.o.y + time * ray.d.y - faces[index].p3.y , ray.o.z + time * ray.d.z - faces[index].p3.z };
		Triple<double> temp = a;

		cross(temp, b);
		cross(b, c);
		cross(c, a);

		double d = dot(temp, b), e = dot(b, c), f = dot(c, temp);

		if (d < 0 || e < 0 || f < 0) {
			times[index] = DBL_MAX;
			return;
		}
		times[index] = time;
	}
}

__global__ void traceRays(Ray* rays, Face* faces, unsigned int numFaces, IntersectionData* data, unsigned int width, unsigned int nRays, unsigned int nReflections) {
	unsigned int index = (blockIdx.x * width + blockIdx.y) * nRays * nRays + threadIdx.x * nRays + threadIdx.y;
	if (faces == NULL) {
		data[index * nReflections].angle = 0.0;
		data[index * nReflections].spectrum = NULL;
		return;
	}
	unsigned int numBlocks = (numFaces & 511 ? (numFaces >> 9) + 1 : numFaces >> 9);
	double* times = new double[numFaces];
	testFaces<<<numBlocks, 512 >>>(rays[index], faces, numFaces, times);
	cudaDeviceSynchronize();
	unsigned int s = smallestTime(times, numFaces);
	if (times[s] < DBL_MAX) {
		rays[index].o.x += rays[index].d.x * times[s];
		rays[index].o.y += rays[index].d.y * times[s];
		rays[index].o.z += rays[index].d.z * times[s];
		data[index * nReflections].spectrum = faces[s].spd;
		data[index * nReflections].angle = dot(rays[index].d, faces[s].n) / (rays[index].d.x * rays[index].d.x + rays[index].d.y * rays[index].d.y + rays[index].d.z * rays[index].d.z);
		double mag = rays[index].d.x * rays[index].d.x + rays[index].d.y * rays[index].d.y + rays[index].d.z * rays[index].d.z;
		rays[index].d.x = rays[index].d.x / mag;
		rays[index].d.y = rays[index].d.y / mag;
		rays[index].d.z = rays[index].d.z / mag;
		mag = 2.0 * dot(rays[index].d, faces[s].n);
		rays[index].d.x -= mag * faces[s].n.x;
		rays[index].d.y -= mag * faces[s].n.y;
		rays[index].d.z -= mag * faces[s].n.z;
		rays[index].index = s;
	}
	delete[] times;
}

__global__ void addToSample(double angle, double* spectrum, double* sample) {
	if (angle < 0 && spectrum[threadIdx.x]) {
		sample[threadIdx.x] -= spectrum[threadIdx.x] * angle;
	}
}

__global__ void computeSamples(IntersectionData* data, double* samples, unsigned int width, unsigned int nRays, unsigned int nReflections) {
	cudaError_t cudaStatus;
	unsigned int index = (blockIdx.x * width + blockIdx.y) * nRays * nRays + threadIdx.x * nRays + threadIdx.y;
	memset(&samples[STEPS * index], 0, sizeof(double) * STEPS);
	for (unsigned int i = 0; i < nReflections; i++) { 
		addToSample << <1, STEPS >> > (data[index * nReflections + i].angle, data[index * nReflections + i].spectrum, &samples[STEPS * index]);
	}
}

__global__ void addToPixel(double* sample, double* pixel, unsigned int nRays) {
	pixel[threadIdx.x] += sample[threadIdx.x] / (nRays * nRays);
}

__global__ void computePixels(double* samples, double* pixels, unsigned int width, unsigned int nRays) {
	unsigned int index = blockIdx.x * width + blockIdx.y;
	memset(&pixels[index * STEPS], 0, sizeof(double) * STEPS);
	for (unsigned int i = 0; i < nRays; i++) {
		for (unsigned int j = 0; j < nRays; j++) {
			addToPixel << <1, STEPS >> > (&samples[((index)*nRays * nRays + i * nRays + j) * STEPS], &pixels[index * STEPS], nRays);
			cudaDeviceSynchronize();
		}
	}
}

__global__ void setSpectrums(Face* faces, unsigned int len, double* spectrums) {
	unsigned int i = (blockIdx.x << 9) + threadIdx.x;
	if (i < len)
		faces[(blockIdx.x << 9) + threadIdx.x].spd = &spectrums[faces[(blockIdx.x << 9) + threadIdx.x].spdIndex];
}