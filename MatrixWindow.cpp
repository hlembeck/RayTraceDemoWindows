#include "MatrixWindow.h"

void handlePaintMatrix(HDC hdc, RECT& rect);

LRESULT CALLBACK MatrixWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	HDC hdc;
	PAINTSTRUCT ps;
	RECT rect;
	switch (message) {
	case WM_CREATE:
		return 0;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		GetClientRect(hWnd, &rect);
		handlePaintMatrix(hdc, rect);
		EndPaint(hWnd, &ps);
		return 0;
	}
	return DefWindowProc(hWnd, message, wParam, lParam);
}


void handlePaintMatrix(HDC hdc, RECT& rect) {

}

HWND createMatrixWindow(HDC hdc) {
	CreateWindow()
}