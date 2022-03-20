#pragma once
#include <windows.h>
#include <vector>

#define ID_EDIT 2

void paintMenu(HDC hdc, RECT& rect);

void addParamWindows(HWND mainWindow, std::vector<HWND>& hWindows, HINSTANCE& hInstance);