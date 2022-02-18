#pragma once

#include "RayTraceEntry.cuh"

#define AppMsg_UpdateDiag WM_APP+1

LRESULT CALLBACK DiagWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);