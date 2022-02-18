#pragma once

#include "RayTraceEntry.cuh"

#define AppMsg_MatrixWindow WM_APP+2
#define MatrixEditID 100

LRESULT CALLBACK MatrixWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);