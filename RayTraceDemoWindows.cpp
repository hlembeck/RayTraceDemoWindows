#include "RayTraceDemoWindows.h"

#define ID_EDIT 1
#define AppMsg_TFINSH WM_APP

//Declarations of functions
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
DWORD WINAPI ThreadProcRT(LPVOID lpParameter);
void printImgParamPinhole(ImgParamPinhole& params);

const Scene scene = { NULL, 0 };

//Vector of window handles
static std::vector<HWND> hWindows;

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd) {
	static TCHAR szAppName[] = TEXT("Ray Tracing Demo");
	MSG msg = {};
	WNDCLASS baseWndClass = { CS_HREDRAW | CS_VREDRAW , WndProc, 0, 0, hInstance, NULL, LoadCursor(NULL, IDC_ARROW), (HBRUSH)GetStockObject(WHITE_BRUSH), NULL, szAppName };

	if (!RegisterClass(&baseWndClass)) {
		MessageBox(NULL, TEXT("Could not register the base window class."), szAppName, MB_ICONERROR);
		return -1;
	}

	hWindows.push_back(CreateWindow(szAppName, szAppName, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)), hInstance, NULL));

	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 30, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 60, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 90, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));

	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP | ES_NUMBER, 150, 150, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP | ES_NUMBER, 150, 180, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));

	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 240, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 270, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));

	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP | ES_NUMBER, 150, 330, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP | ES_NUMBER, 150, 360, 100, 20, hWindows[0], (HMENU)ID_EDIT, hInstance, NULL));

	hWindows.push_back(CreateWindow(TEXT("BUTTON"), TEXT("SUBMIT"), WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 50, 390, 100, 20, hWindows[0], NULL, hInstance, NULL));

	ShowWindow(hWindows[0], nShowCmd);
	UpdateWindow(hWindows[0]);

	while (GetMessage(&msg, NULL, 0, 0))
	{
		if (!IsDialogMessage(hWindows[0],&msg)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}
	return msg.wParam;

	return 0;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	HDC hdc;
	PAINTSTRUCT ps;
	RECT rect;
	int i;
	HWND nWnd;
	ImgParamPinhole params;
	TCHAR buf[288] = {};
	TCHAR* endPtr;
	static RTParams paramsRT = {};
	static HANDLE hThread = 0;
	DWORD exitCode;

	switch (message) {
	case WM_CREATE:
		return 0;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		GetClientRect(hWnd, &rect);

		paintMenu(hdc, rect);

		EndPaint(hWnd, &ps);
		return 0;
	case WM_COMMAND:
		switch (wParam) {
		case BN_CLICKED:
			endPtr = 0;
			i = 0;
			memset(buf, 0, 576);
			SendMessage(hWindows[1], WM_GETTEXT, 32, (LPARAM)buf);
			params.pinhole.x = wcstod(buf,&endPtr);
			SendMessage(hWindows[2], WM_GETTEXT, 32, (LPARAM)endPtr);
			params.pinhole.y = wcstod(endPtr, &endPtr);
			SendMessage(hWindows[3], WM_GETTEXT, 32, (LPARAM)endPtr);
			params.pinhole.z = wcstod(endPtr, &endPtr);

			SendMessage(hWindows[4], WM_GETTEXT, 32, (LPARAM)endPtr);
			params.width = wcstol(endPtr, &endPtr, 10);
			SendMessage(hWindows[5], WM_GETTEXT, 32, (LPARAM)endPtr);
			params.height = wcstol(endPtr, &endPtr, 10);

			SendMessage(hWindows[6], WM_GETTEXT, 32, (LPARAM)endPtr);
			params.sensorWidth = wcstod(endPtr, &endPtr);
			SendMessage(hWindows[7], WM_GETTEXT, 32, (LPARAM)endPtr);
			params.sensorHeight = wcstod(endPtr, &endPtr);

			SendMessage(hWindows[8], WM_GETTEXT, 32, (LPARAM)endPtr);
			params.nRays = wcstol(endPtr, &endPtr, 10);
			SendMessage(hWindows[9], WM_GETTEXT, 32, (LPARAM)endPtr);
			params.nReflections = wcstol(endPtr, &endPtr, 10);

			paramsRT.params = params;
			if(!hThread)
				hThread = CreateThread(NULL, 0, ThreadProcRT, &paramsRT, 0, NULL);
		}
		return 0;
	case AppMsg_TFINSH:
		WaitForSingleObject(hThread, INFINITE);
		exitCode = 0;
		if (GetExitCodeThread(hThread, &exitCode)) {
			CloseHandle(hThread);
			//TODO: Write bitmap image on disk to screen
		}
		else
			CloseHandle(hThread);
		return 0;
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hWnd, message, wParam, lParam);
}

void printImgParamPinhole(ImgParamPinhole& params) {
	TCHAR* buf = new TCHAR[1028];
	swprintf(buf, L"Pinhole: (%f, %f, %f)\nWidth: %d\nHeight: %d\nSensor Width: %f\nSensor Height: %f\nNum Rays: %d\nNum Reflections: %d\n", params.pinhole.x, params.pinhole.y, params.pinhole.z, params.width, params.height, params.sensorWidth, params.sensorHeight, params.nRays, params.nReflections);
	OutputDebugString(buf);
	delete[] buf;
}

DWORD WINAPI ThreadProcRT(LPVOID lpParameter) {
	ImgParamPinhole params = ((RTParams*)lpParameter)->params;
	unsigned char* rgbQuadArr = ((RTParams*)lpParameter)->rgbQuadArr;
	if (!getPinholeImage(params, ((RTParams*)lpParameter)->scene, rgbQuadArr)) {
		OutputDebugString(TEXT("(In ThreadProcRT()) Failed getPinholeBitmap().\n"));
		return 1;
	}
	if (!savePNG(rgbQuadArr, params.width, params.height, "image.png")) {
		OutputDebugString(TEXT("(In ThreadProcRT()) Failed savePNG().\n"));
		return 1;
	}
	return 0;
}