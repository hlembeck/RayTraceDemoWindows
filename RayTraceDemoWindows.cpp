#include "RayTraceDemoWindows.h"

#define ID_EDIT 1

//Declarations of functions
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK WndProcRT(HWND, UINT, WPARAM, LPARAM);
bool handleButtonClick(LPARAM lParam);
DWORD WINAPI ThreadProcRT(LPVOID lpParameter);

//Vector of window handles
static std::vector<HWND> hWindows;

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd) {
	static TCHAR szAppName[] = TEXT("Ray Tracing Demo");
	MSG msg = {};
	WNDCLASS baseWndClass = { CS_HREDRAW | CS_VREDRAW , WndProc, 0, 0, hInstance, NULL, LoadCursor(NULL, IDC_ARROW), (HBRUSH)GetStockObject(WHITE_BRUSH), NULL, szAppName };

	WNDCLASS imageWndClass = { CS_GLOBALCLASS | CS_OWNDC , WndProcRT, 0, 0, hInstance, NULL, LoadCursor(NULL, IDC_ARROW), (HBRUSH)GetStockObject(WHITE_BRUSH), NULL, TEXT("Image Window Class") };

	if (!RegisterClass(&baseWndClass)) {
		MessageBox(NULL, TEXT("Could not register the base window class."), szAppName, MB_ICONERROR);
		return -1;
	}

	if (!RegisterClass(&imageWndClass)) {
		MessageBox(NULL, TEXT("Could not register the image window class."), TEXT("Image Window Class"), MB_ICONERROR);
		return -1;
	}

	hWindows.push_back(CreateWindow(szAppName, szAppName, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, 400, 800, NULL, LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)), hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 50, 30, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 50, 60, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 50, 90, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("BUTTON"), TEXT("SUBMIT"), WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 50, 120, 100, 20, hWindows[0], NULL, hInstance, NULL));

	ShowWindow(hWindows[0], nShowCmd);
	UpdateWindow(hWindows[0]);

	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return msg.wParam;

	return 0;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	//static HWND hWndEdit;
	static RECT pinholeRect = { 0, 30, 50, 110 };
	HDC hdc;
	PAINTSTRUCT ps;
	RECT rect;

	switch (message) {
	case WM_CREATE:
		return 0;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		GetClientRect(hWnd, &rect);
		DrawText(hdc, TEXT("Coordinates of Pinhole"), -1, &rect, DT_SINGLELINE | DT_LEFT | DT_TOP);
		DrawText(hdc, TEXT("X:"), -1, &pinholeRect, DT_SINGLELINE | DT_CENTER | DT_TOP);
		DrawText(hdc, TEXT("Y:"), -1, &pinholeRect, DT_SINGLELINE | DT_CENTER | DT_VCENTER);
		DrawText(hdc, TEXT("Z:"), -1, &pinholeRect, DT_SINGLELINE | DT_CENTER | DT_BOTTOM);

		/*DrawText(hdc, TEXT("Image Parameters"), -1, &rect, DT_SINGLELINE | DT_LEFT | DT_TOP);
		DrawText(hdc, TEXT("Width:"), -1, &pinholeRect, DT_SINGLELINE | DT_CENTER | DT_TOP);
		DrawText(hdc, TEXT("Height:"), -1, &pinholeRect, DT_SINGLELINE | DT_CENTER | DT_VCENTER);
		DrawText(hdc, TEXT("Sensor Width:"), -1, &pinholeRect, DT_SINGLELINE | DT_CENTER | DT_VCENTER);
		DrawText(hdc, TEXT("Sensor Height:"), -1, &pinholeRect, DT_SINGLELINE | DT_CENTER | DT_VCENTER);
		DrawText(hdc, TEXT("Rays/Pixel:"), -1, &pinholeRect, DT_SINGLELINE | DT_CENTER | DT_VCENTER);
		DrawText(hdc, TEXT("Reflections/Ray:"), -1, &pinholeRect, DT_SINGLELINE | DT_CENTER | DT_VCENTER);*/
		EndPaint(hWnd, &ps);
		return 0;
	case WM_COMMAND:
		switch (HIWORD(wParam)) {
		case BN_CLICKED:
			if (!handleButtonClick(lParam))
				MessageBox(NULL, TEXT("No Button Found."), TEXT("Notification"), MB_OK);
		}
		return 0;
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hWnd, message, wParam, lParam);
}

LRESULT CALLBACK WndProcRT(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	HDC hdc, hdcBMP;
	PAINTSTRUCT ps;
	RECT rect;
	static HANDLE hThread;
	static RTParams* paramsRT;

	switch (message) {
	case WM_CREATE:
		paramsRT = (RTParams*)(((CREATESTRUCT*)lParam)->lpCreateParams);
		hThread = CreateThread(NULL, 0, ThreadProcRT, ((CREATESTRUCT*)lParam)->lpCreateParams, 0, NULL);
		if (hThread) {
			WaitForSingleObject(hThread, INFINITE);
			CloseHandle(hThread);
			delete[] paramsRT->rgbQuadArr;
		}
		return 0;
	case WM_PAINT:
		/*WaitForSingleObject(hThread, INFINITE);
		CloseHandle(hThread);*/
		hdc = BeginPaint(hWnd, &ps);
		GetClientRect(hWnd, &rect);
		
		EndPaint(hWnd, &ps);
		return 0;
	case WM_DESTROY:
		OutputDebugString(TEXT("TEST\n"));
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hWnd, message, wParam, lParam);
}

bool handleButtonClick(LPARAM lParam) {
	TCHAR buf[150] = { };
	static Triple<double> pinhole = { 0.0, 0.0, 1.0 };
	unsigned int width = 256;
	unsigned int height = 256;
	double sensorWidth = 2;
	double sensorHeight = 2;
	unsigned int nRays = 1;
	unsigned int nReflections = 1;

	//Scene must be a parameter of paramsRT
	RTParams paramsRT;
	paramsRT.scene = { NULL,0 };

	if ((HWND)lParam == hWindows[4]) {
		SendMessage(hWindows[1], WM_GETTEXT, 50, (LPARAM)&buf);
		SendMessage(hWindows[2], WM_GETTEXT, 50, (LPARAM)&buf[50]);
		SendMessage(hWindows[3], WM_GETTEXT, 50, (LPARAM)&buf[100]);
		try { pinhole.x = std::stod((TCHAR*)buf); }
		catch (...) { pinhole.x = 0.0; }
		try { pinhole.y = std::stod((TCHAR*)&buf[50]); }
		catch (...) { pinhole.y = 0.0; }
		try { pinhole.z = std::stod((TCHAR*)&buf[100]); }
		catch (...) { pinhole.z = 1.0; }
		paramsRT.params = { pinhole, width, height, sensorWidth, sensorHeight, nRays, nReflections };
		HBITMAP hBMP = {};

		hWindows.push_back(CreateWindow(TEXT("Image Window Class"), TEXT("RT Image Window"), WS_OVERLAPPED | WS_VISIBLE, 0, 0, paramsRT.params.width, paramsRT.params.height, hWindows[0], NULL, NULL, &paramsRT));
		if (hWindows[5] == NULL)
			OutputDebugString(TEXT("Failed to create window"));
		return true;
	}
	return false;
}

DWORD WINAPI ThreadProcRT(LPVOID lpParameter) {
	ImgParamPinhole params = ((RTParams*)lpParameter)->params;
	if (!getPinholeBitmap(params, ((RTParams*)lpParameter)->scene, ((RTParams*)lpParameter)->rgbQuadArr)) {
		OutputDebugString(TEXT("(In ThreadProcRT()) Failed getPinholeBitmap().\n"));
		return 0;
	}
	return 0;
}