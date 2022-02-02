#include "SceneTransform.cuh"

__device__ void crossST(Triple<double>& v, Triple<double>& w) {
	double a, b;
	a = v.y * w.z - w.z * v.y;
	b = v.z * w.x - v.x * w.z;
	v.z = v.x * w.y - v.y * w.x;
	v.x = a;
	v.y = b;
}

double* getTranslateTransform(Triple<double>& p, cudaError_t& cudaStatus) {
	double* ret, * temp = new double[16];
	memset(temp, 0, sizeof(double) * 15);
	temp[0] = 1.0;
	temp[3] = p.x;
	temp[5] = 1.0;
	temp[7] = p.y;
	temp[10] = 1.0;
	temp[11] = p.z;
	temp[15] = 1.0;

	cudaMalloc((void**)&ret, sizeof(double) * 16);
	cudaStatus = cudaMemcpy(ret, temp, sizeof(double) * 16, cudaMemcpyHostToDevice);
	delete[] temp;
	return ret;
}

double* getScaleTransform(Triple<double>& v, cudaError_t& cudaStatus) {
	double* ret, * temp = new double[16];
	memset(temp, 0, sizeof(double) * 15);
	temp[0] = v.x;
	temp[5] = v.y;
	temp[10] = v.z;
	temp[15] = 1.0;

	cudaMalloc((void**)&ret, sizeof(double) * 16);
	cudaStatus = cudaMemcpy(ret, temp, sizeof(double) * 16, cudaMemcpyHostToDevice);
	delete[] temp;
	return ret;
}

void normalize(Triple<double>& v) {
	double mag = v.x * v.x + v.y * v.y + v.z * v.z;
	if (mag == 0)
		return;
	v.x = v.x / mag;
	v.y = v.y / mag;
	v.z = v.z / mag;
}

double* getRotateTransform(Triple<double>& v, double a, cudaError_t& cudaStatus) {
	double* ret, * temp = new double[16];
	double c, s;
	c = cos(a);
	s = sin(a);
	memset(temp, 0, sizeof(double) * 15);
	temp[0] = v.x * v.x + (1 - v.x * v.x) * c;
	temp[1] = v.x * v.y * (1 - c) - v.z * s;
	temp[2] = v.x * v.z * (1 - c) + v.y * s;

	temp[4] = v.y * v.x * (1 - c) + v.z * s;
	temp[5] = v.y * v.y + (1 - v.y * v.y) * c;
	temp[6] = v.y * v.z * (1 - c) - v.x * s;

	temp[8] = v.z * v.x * (1 - c) - v.y * s;
	temp[9] = v.z * v.y * (1 - c) + v.x * s;
	temp[10] = v.z * v.z + (1 - v.z * v.z) * c;

	temp[15] = 1;

	cudaMalloc((void**)&ret, sizeof(double) * 16);
	cudaStatus = cudaMemcpy(ret, temp, sizeof(double) * 16, cudaMemcpyHostToDevice);
	delete[] temp;
	return ret;
}

__device__ void transformPoint(Triple<double>& p, double* mat) {
	double a;
	double b;
	a = mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3];
	b = mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7];
	p.z = mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11];
	p.y = b;
	p.x = a;
}

__global__ void transformFaces(Face* faces, double* mat, unsigned int len) {
	unsigned int index = blockIdx.x * 512 + threadIdx.x;
	if (index < len) {
		transformPoint(faces[index].p1, mat);
		transformPoint(faces[index].p2, mat);
		transformPoint(faces[index].p3, mat);
		faces[index].n = { faces[index].p2.x - faces[index].p1.x, faces[index].p2.y - faces[index].p1.y, faces[index].p2.z - faces[index].p1.z };
		Triple<double> a = { faces[index].p3.x - faces[index].p2.x, faces[index].p3.y - faces[index].p2.y, faces[index].p3.z - faces[index].p2.z };
		crossST(faces[index].n, a);
		double mag = sqrt(faces[index].n.x * faces[index].n.x + faces[index].n.y * faces[index].n.y + faces[index].n.z * faces[index].n.z);
		faces[index].n.x = faces[index].n.x / mag;
		faces[index].n.y = faces[index].n.y / mag;
		faces[index].n.z = faces[index].n.z / mag;
	}
}

__global__ void transformMeshes(Mesh* meshes, double* mat) {
	unsigned int len = meshes[blockIdx.x].len;
	unsigned int numBlocks = (len & 511 ? (len >> 9) + 1 : len >> 9);
	transformFaces << <numBlocks, 512 >> > (meshes[blockIdx.x].faces, mat, len);
}