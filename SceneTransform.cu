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

double* getRotateTransformHOST(Triple<double>& v, double a) {
	double* ret = new double[16];
	double c, s;
	c = cos(a);
	s = sin(a);
	memset(ret, 0, sizeof(double) * 15);
	ret[0] = v.x * v.x + (1 - v.x * v.x) * c;
	ret[1] = v.x * v.y * (1 - c) - v.z * s;
	ret[2] = v.x * v.z * (1 - c) + v.y * s;

	ret[4] = v.y * v.x * (1 - c) + v.z * s;
	ret[5] = v.y * v.y + (1 - v.y * v.y) * c;
	ret[6] = v.y * v.z * (1 - c) - v.x * s;

	ret[8] = v.z * v.x * (1 - c) - v.y * s;
	ret[9] = v.z * v.y * (1 - c) + v.x * s;
	ret[10] = v.z * v.z + (1 - v.z * v.z) * c;

	ret[15] = 1;
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

void setMeshParams(Face* faces, unsigned int len, double refractiveIndex, double reflectivity) {
	while (len) {
		faces[--len].reflectivity = reflectivity;
		faces[len].refractiveIndex = refractiveIndex;
	}
}

void crossProductHOST(Triple<double>& v1, Triple<double>& v2, Triple<double>& res) {
	res.x = v1.y * v2.z - v1.z * v2.y;
	res.y = v1.z * v2.x - v1.x * v2.z;
	res.z = v1.x * v2.y - v1.y * v2.x;
}

void transformFaceHOST(Face& face, double* mat) {
	double a, b;
	a = mat[0] * face.p1.x + mat[1] * face.p1.y + mat[2] * face.p1.z + mat[3];
	b = mat[4] * face.p1.x + mat[5] * face.p1.y + mat[6] * face.p1.z + mat[7];
	face.p1.z = mat[8] * face.p1.x + mat[9] * face.p1.y + mat[10] * face.p1.z + mat[11];
	face.p1.x = a;
	face.p1.y = b;

	a = mat[0] * face.p2.x + mat[1] * face.p2.y + mat[2] * face.p2.z + mat[3];
	b = mat[4] * face.p2.x + mat[5] * face.p2.y + mat[6] * face.p2.z + mat[7];
	face.p2.z = mat[8] * face.p2.x + mat[9] * face.p2.y + mat[10] * face.p2.z + mat[11];
	face.p2.x = a;
	face.p2.y = b;

	a = mat[0] * face.p3.x + mat[1] * face.p3.y + mat[2] * face.p3.z + mat[3];
	b = mat[4] * face.p3.x + mat[5] * face.p3.y + mat[6] * face.p3.z + mat[7];
	face.p3.z = mat[8] * face.p3.x + mat[9] * face.p3.y + mat[10] * face.p3.z + mat[11];
	face.p3.x = a;
	face.p3.y = b;

	Triple<double> v1{face.p2.x - face.p1.x , face.p2.y - face.p1.y , face.p2.z - face.p1.z};
	Triple<double> v2{face.p3.x - face.p2.x , face.p3.y - face.p2.y , face.p3.z - face.p2.z};
	crossProductHOST(v1, v2, face.n);
	a = sqrt(face.n.x * face.n.x + face.n.y * face.n.y + face.n.z * face.n.z);
	if (a) {
		face.n.x /= a;
		face.n.y /= a;
		face.n.z /= a;
	}
}

void transformMeshHOST(Face* faces, unsigned int len, double* mat) {
	for (unsigned int i = 0; i < len; i++) {
		transformFaceHOST(faces[i], mat);
	}
}

void multiply(double* a, double*& b) {
	double* temp = new double[16];
	temp[0] = a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12];
	temp[1] = a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13];
	temp[2] = a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14];
	temp[3] = a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15];

	temp[4] = a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12];
	temp[5] = a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13];
	temp[6] = a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14];
	temp[7] = a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15];

	temp[8] = a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12];
	temp[9] = a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13];
	temp[10] = a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14];
	temp[11] = a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15];

	temp[12] = a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12];
	temp[13] = a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13];
	temp[14] = a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14];
	temp[15] = a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15];

	delete[] b;
	b = temp;
}