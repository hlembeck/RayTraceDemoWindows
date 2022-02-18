#pragma once
#include <windows.h>
#include <vector>

#define ID_EDIT 1

void paintMenu(HDC hdc, RECT& rect);

void addParamWindows(std::vector<HWND>& hWindows, HINSTANCE& hInstance);