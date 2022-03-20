#pragma once

#include "RayTraceEntry.cuh"

#define AppMsg_MatrixWindow WM_APP+2
#define MatrixEditID 100

static const TCHAR szMatrixName[] = L"Matrix Window";
static double matrix[16] = {};

LRESULT CALLBACK MatrixWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

windowInfo createMatrixWindow(HWND hWndMain, HINSTANCE hInstance);