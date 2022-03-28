#pragma once

#include "BasicTypes.h"

#define AppMsg_TransformationWindow WM_APP + 32
#define transformationMenuID 256

static const TCHAR szTransformationWndName[] = L"Transformation Window";

LRESULT CALLBACK TransformationWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

windowInfo createTransformationWindow(HWND hWndMain, HINSTANCE hInstacnce);