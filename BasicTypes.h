#pragma once
#include <string>
#include <math.h>
#include <limits>
#include <vector>
#include <windows.h>

constexpr unsigned int STEPS = 100;
constexpr unsigned int MIN_WAVELENGTH = 350;
constexpr unsigned int MAX_WAVELENGTH = 750;

template <typename T> struct Triple {
	T x;
	T y;
	T z;
};

struct ImgParamPinhole {
	Triple<double> pinhole;
	unsigned int width;
	unsigned int height;
	double sensorWidth;
	double sensorHeight;
	unsigned int nRays;
	unsigned int nReflections;
};

struct Face {
	Triple<double> p1;
	Triple<double> p2;
	Triple<double> p3;
	Triple<double> n;
	union {
		double* spd;
		unsigned int spdIndex;
	};
};

struct Mesh {
	Face* faces;
	unsigned int len;
};

struct Scene {
	Mesh* meshes;
	unsigned int len;
};

struct Ray {
	Triple<double> o;
	Triple<double> d;
};

struct IntersectionData {
	double* spectrum;
	double angle;
};

struct windowInfo {
	HWND hwnd;
	std::vector<HWND>& children;
};