#include "MatrixWindow.h"

static HWND hMainWnd;
static HWND hMatrixWnd;

static std::vector<HWND> hWindowsMatrix;

LRESULT CALLBACK MatrixWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	HDC hdc;
	PAINTSTRUCT ps;
	RECT rect;
	TCHAR *buf, *tBuf;
	unsigned int bufIndex = 0;
	switch (message) {
	case WM_CREATE:
		return 0;
	case WM_COMMAND:
		switch (HIWORD(wParam)) {
		case BN_CLICKED:
			buf = new TCHAR[512];
			tBuf = buf;
			memset(buf, 0, sizeof(TCHAR) * 512);
			for (unsigned int i = 0; i < 16; i++) {
				bufIndex += SendMessage(hWindowsMatrix[i], WM_GETTEXT, 16, (LPARAM)(tBuf)) + 1;
				matrix[i] = wcstod(tBuf, NULL);
				tBuf += bufIndex;
			}
			delete[] buf;
			SendMessage(hMatrixWnd, WM_CLOSE, NULL, NULL);
			SendMessage(hMainWnd, AppMsg_MatrixWindow, (WPARAM)matrix, NULL);
			return 0;
		}
		break;
	case WM_CLOSE:
		hWindowsMatrix.clear();
	}

	return DefWindowProc(hWnd, message, wParam, lParam);
}

windowInfo createMatrixWindow(HWND hWndMain, HINSTANCE hInstance) {
	hMainWnd = hWndMain;
	windowInfo ret = {0,hWindowsMatrix};

	//Parent Window
	ret.hwnd = CreateWindow(
		szMatrixName,
		szMatrixName,
		WS_EX_CONTROLPARENT | WS_OVERLAPPED | WS_SYSMENU,
		0,
		0,
		200,
		200,
		NULL,
		NULL,
		hInstance,
		NULL
	);

	//Matrix entry windows
	for (unsigned int i = 0; i < 16; i++) {
		hWindowsMatrix.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			(i % 4) * 40,
			(i / 4) * 20,
			40,
			20,
			ret.hwnd,
			(HMENU)MatrixEditID + i,
			hInstance,
			NULL
		));
	}

	hWindowsMatrix.push_back(CreateWindow(
		TEXT("BUTTON"),
		TEXT("SUBMIT"),
		WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		0,
		80,
		160,
		20,
		ret.hwnd,
		(HMENU)MatrixEditID + 16,
		hInstance,
		NULL
	));

	hMatrixWnd = ret.hwnd;

	ShowWindow(ret.hwnd, SW_SHOW);
	UpdateWindow(ret.hwnd);
	return ret;
}