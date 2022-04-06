#include "RayTraceDemoWindows.h"

#define AppMsg_TFINISH WM_APP

//Declarations of functions
bool initializeWindows(HINSTANCE hInstance, int nShowCmd);
void spectrumFromRGB(COLORREF rgbColor, std::vector<double>& spectrums);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
DWORD WINAPI ThreadProcRT(LPVOID lpParameter);
void printImgParamPinhole(ImgParamPinhole& params);
void deleteSceneParams();
bool displayImage(unsigned char* rgbQuadArr, unsigned int cx, unsigned int cy);

const TCHAR szDiagName[] = TEXT("Diagnostic Window");

static SceneParams sceneParams = {};

static CHOOSECOLOR chooseColorStruct = {};
static COLORREF acrCustClr[16];

//Vector of window handles
static std::vector<HWND> hWindows;
static HWND diagnosticWnd = 0;
static std::vector<windowInfo> windowManager;

inline void popWindow() {
	for (int i = 0; i < windowManager.back().children.size(); i++) {
		DestroyWindow(windowManager.back().children[i]);
	}
	windowManager.back().children.clear();
	DestroyWindow(windowManager.back().hwnd);
	windowManager.pop_back();
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd) {
	MSG msg = {};

	if (!initializeWindows(hInstance, nShowCmd))
		return 1;

	while (GetMessage(&msg, NULL, 0, 0))
	{
		for (unsigned int i = 0; i < windowManager.size(); i++) {
			if (IsDialogMessage(windowManager[i].hwnd,&msg))
				goto next;
		}
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	next:
		continue;
	}

	return 0;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	HDC hdc;
	PAINTSTRUCT ps;
	RECT rect;
	int i;
	HWND nWnd;
	ImgParamPinhole params;
	TCHAR buf[512] = {};
	TCHAR* endPtr;
	//Used to transform meshes before they are added to faceVector above.
	static double* transformMatrix;

	static RTParams paramsRT = { {}, sceneParams, NULL };
	static HANDLE hThread = 0;
	DWORD exitCode;

	switch (message) {
	case WM_CREATE:
		AllocConsole();
		FILE* fDummy;
		freopen_s(&fDummy, "CONIN$", "r", stdin);
		freopen_s(&fDummy, "CONOUT$", "w", stderr);
		freopen_s(&fDummy, "CONOUT$", "w", stdout);
		transformMatrix = new double[16];
		memset(transformMatrix, 0, sizeof(double) * 16);
		transformMatrix[0] = 1.0;
		transformMatrix[5] = 1.0;
		transformMatrix[10] = 1.0;
		transformMatrix[15] = 1.0;
		return 0;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		GetClientRect(hWnd, &rect);

		paintMenu(hdc, rect);

		EndPaint(hWnd, &ps);
		return 0;
	case WM_COMMAND:
		switch (HIWORD(wParam)) {
		case BN_CLICKED:
			switch (LOWORD(wParam)) {
			case 0:
				chooseColorStruct = {
					sizeof(CHOOSECOLOR),
					windowManager[0].hwnd,
					NULL,
					0,
					acrCustClr,
					CC_FULLOPEN,
					NULL,
					NULL,
					NULL
				};
				if (ChooseColor(&chooseColorStruct)) {
					windowManager.push_back(createTransformationWindow(hWnd, GetModuleHandle(NULL)));

					addPrismHOST(sceneParams.faces, transformMatrix, sceneParams.spectrums.size());
					spectrumFromRGB(chooseColorStruct.rgbResult, sceneParams.spectrums);
					sceneParams.meshes.push_back(sceneParams.faces.size());
				}
				return 0;
			case 1:
				chooseColorStruct = {
					sizeof(CHOOSECOLOR),
					windowManager[0].hwnd,
					NULL,
					0,
					acrCustClr,
					CC_FULLOPEN,
					NULL,
					NULL,
					NULL
				};
				if (ChooseColor(&chooseColorStruct)) {
					windowManager.push_back(createTransformationWindow(hWnd, GetModuleHandle(NULL)));
					addPlateHOST(sceneParams.faces, transformMatrix,sceneParams.spectrums.size());
					spectrumFromRGB(chooseColorStruct.rgbResult, sceneParams.spectrums);
					sceneParams.meshes.push_back(sceneParams.faces.size());
				}
				return 0;
			case ID_EDIT + 9:
				endPtr = 0;
				i = 0;
				memset(buf, 0, 1024);
				SendMessage(hWindows[2], WM_GETTEXT, 32, (LPARAM)buf);
				params.pinhole.x = wcstod(buf, &endPtr);
				SendMessage(hWindows[3], WM_GETTEXT, 32, (LPARAM)endPtr);
				params.pinhole.y = wcstod(endPtr, &endPtr);
				SendMessage(hWindows[4], WM_GETTEXT, 32, (LPARAM)endPtr);
				params.pinhole.z = wcstod(endPtr, &endPtr);

				SendMessage(hWindows[5], WM_GETTEXT, 32, (LPARAM)endPtr);
				params.width = wcstol(endPtr, &endPtr, 10);
				SendMessage(hWindows[6], WM_GETTEXT, 32, (LPARAM)endPtr);
				params.height = wcstol(endPtr, &endPtr, 10);

				SendMessage(hWindows[7], WM_GETTEXT, 32, (LPARAM)endPtr);
				params.sensorWidth = wcstod(endPtr, &endPtr);
				SendMessage(hWindows[8], WM_GETTEXT, 32, (LPARAM)endPtr);
				params.sensorHeight = wcstod(endPtr, &endPtr);

				SendMessage(hWindows[9], WM_GETTEXT, 32, (LPARAM)endPtr);
				params.nRays = wcstol(endPtr, &endPtr, 10);
				SendMessage(hWindows[10], WM_GETTEXT, 32, (LPARAM)endPtr);
				params.nReflections = wcstol(endPtr, &endPtr, 10);

				paramsRT.params = params;
				if (!hThread) {
					if (params.width == 0 && params.height == 0)
						displayImage(NULL,0,0);
					else
						hThread = CreateThread(NULL, 0, ThreadProcRT, &paramsRT, 0, NULL);
				}
				return 0;
			}
		}
		return 0;
	case AppMsg_MatrixWindow:
		popWindow();
		if (sceneParams.meshes.size() < 2)
			transformMeshHOST(sceneParams.faces.data(), sceneParams.meshes.back(), (double*)wParam);
		else {
			i = sceneParams.meshes[sceneParams.meshes.size() - 2];
			transformMeshHOST(&sceneParams.faces[i], sceneParams.meshes.back() - i, (double*)wParam);
		}
		if (!IsWindow(diagnosticWnd)) {
			diagnosticWnd = CreateWindow(szDiagName, szDiagName, WS_OVERLAPPEDWINDOW | WS_VSCROLL, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, GetModuleHandle(NULL), NULL);

			ShowWindow(diagnosticWnd, SW_SHOW);
		}
		SendMessage(diagnosticWnd, AppMsg_UpdateDiag, (WPARAM)&sceneParams, (LPARAM)&chooseColorStruct.rgbResult);
		return 0;
	case AppMsg_TransformationWindow:
		switch (wParam) {
		case 0:
			SendMessage(windowManager.back().hwnd, WM_CLOSE, 0, 0);
			popWindow();
			//windowManager.push_back(createTranslateWindow(windowManager[0].hwnd, GetModuleHandle(NULL)));
			break;
		case 1:
			SendMessage(windowManager.back().hwnd, WM_CLOSE, 0, 0);
			popWindow();
			//windowManager.push_back(createScaleWindow(windowManager[0].hwnd, GetModuleHandle(NULL)));
			break;
		case 2:
			SendMessage(windowManager.back().hwnd, WM_CLOSE, 0, 0);
			popWindow();
			//windowManager.push_back(createRotateWindow(windowManager[0].hwnd, GetModuleHandle(NULL)));
			break;
		case 3:
			SendMessage(windowManager.back().hwnd, WM_CLOSE, 0, 0);
			windowManager.back().children.clear();
			popWindow();
			windowManager.push_back(createMatrixWindow(windowManager[0].hwnd, GetModuleHandle(NULL)));
		}
		return 0;
	case AppMsg_TransformationWindow + 1:
		popWindow();
		if (sceneParams.meshes.size() < 2)
			transformMeshHOST(sceneParams.faces.data(), sceneParams.meshes.back(), (double*)wParam);
		else {
			i = sceneParams.meshes[sceneParams.meshes.size() - 2];
			transformMeshHOST(&sceneParams.faces[i], sceneParams.meshes.back() - i, (double*)wParam);
		}
		delete[] (double*)wParam;
		windowManager.push_back(createTransformationWindow(hWnd, GetModuleHandle(NULL)));
		/*if (!IsWindow(diagnosticWnd)) {
			diagnosticWnd = CreateWindow(szDiagName, szDiagName, WS_OVERLAPPEDWINDOW | WS_VSCROLL, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, GetModuleHandle(NULL), NULL);

			ShowWindow(diagnosticWnd, SW_SHOW);
		}
		SendMessage(diagnosticWnd, AppMsg_UpdateDiag, (WPARAM)&sceneParams, (LPARAM)&chooseColorStruct.rgbResult);*/
		return 0;
	case AppMsg_TFINISH:
		WaitForSingleObject(hThread, INFINITE);
		exitCode = 0;
		if (GetExitCodeThread(hThread, &exitCode)) {
			CloseHandle(hThread);
			//TODO: Write bitmap image on disk to screen
		}
		else
			CloseHandle(hThread);
		hThread = 0;
		return 0;
	case WM_DESTROY:
		FreeConsole();
		delete[] transformMatrix;
		deleteSceneParams();
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hWnd, message, wParam, lParam);
}

bool initializeWindows(HINSTANCE hInstance, int nShowCmd) {
	TCHAR szAppName[] = TEXT("Ray Tracing Demo");
	HBRUSH hBrush = (HBRUSH)GetStockObject(WHITE_BRUSH);
	//Main window class. Used for displaying initial menu and RT-generated bitmaps
	WNDCLASS baseWndClass = {
		CS_HREDRAW | CS_VREDRAW,
		WndProc,
		0,
		0,
		hInstance,
		NULL,
		LoadCursor(NULL, IDC_ARROW),
		hBrush,
		NULL,
		szAppName };
	if (!RegisterClass(&baseWndClass)) {
		MessageBox(NULL, TEXT("Could not register the base window class."), szAppName, MB_ICONERROR);
		return -1;
	}

	//Diagnostic window class. Used for displaying mesh information
	WNDCLASS diagnosticWndClass = {
		CS_HREDRAW | CS_VREDRAW,
		DiagWndProc,
		0,
		0,
		hInstance,
		NULL,
		LoadCursor(NULL, IDC_ARROW),
		hBrush,
		NULL,
		szDiagName
	};
	if (!RegisterClass(&diagnosticWndClass)) {
		MessageBox(NULL, TEXT("Could not register the diagnostics window class."), szDiagName, MB_ICONERROR);
		return -1;
	}

	//Transformation window class. Used when the user wishes to transform parts of the scene.
	WNDCLASS transformationWndClass = {
		CS_HREDRAW | CS_VREDRAW,
		TransformationWndProc,
		0,
		0,
		hInstance,
		NULL,
		LoadCursor(NULL, IDC_ARROW),
		hBrush,
		NULL,
		szTransformationWndName
	};
	if (!RegisterClass(&transformationWndClass)) {
		MessageBox(NULL, TEXT("Could not register the transformation window class."), szDiagName, MB_ICONERROR);
		return -1;
	}

	WNDCLASS matrixWndClass = {
		CS_HREDRAW | CS_VREDRAW,
		MatrixWndProc,
		0,
		0,
		hInstance,
		NULL,
		LoadCursor(NULL,IDC_ARROW),
		hBrush,
		NULL,
		szMatrixName
	};
	if (!RegisterClass(&matrixWndClass)) {
		MessageBox(NULL, TEXT("Could not register the matrix window class."), szDiagName, MB_ICONERROR);
		return -1;
	}

	windowManager.push_back({
		CreateWindow(szAppName, szAppName, WS_EX_CONTROLPARENT | WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)), hInstance, NULL),
		hWindows
		});
	addParamWindows(windowManager[0].hwnd, hWindows, hInstance);

	ShowWindow(windowManager[0].hwnd, nShowCmd);
	UpdateWindow(windowManager[0].hwnd);
}

void printImgParamPinhole(ImgParamPinhole& params) {
	TCHAR* buf = new TCHAR[1028];
	swprintf(buf, L"Pinhole: (%f, %f, %f)\nWidth: %d\nHeight: %d\nSensor Width: %f\nSensor Height: %f\nNum Rays: %d\nNum Reflections: %d\n", params.pinhole.x, params.pinhole.y, params.pinhole.z, params.width, params.height, params.sensorWidth, params.sensorHeight, params.nRays, params.nReflections);
	OutputDebugString(buf);
	delete[] buf;
}

DWORD WINAPI ThreadProcRT(LPVOID lpParameter) {
	ImgParamPinhole& params = ((RTParams*)lpParameter)->params;
	unsigned char*& rgbQuadArr = ((RTParams*)lpParameter)->rgbQuadArr;
	if (!getPinholeImage(*(RTParams*)lpParameter)) {
		OutputDebugString(TEXT("(In ThreadProcRT()) Failed getPinholeBitmap().\n"));
		delete[] rgbQuadArr;
		rgbQuadArr = 0;
		return 1;
	}

	if (!displayImage(rgbQuadArr,params.width,params.height)) {
		OutputDebugString(TEXT("(In ThreadProcRT()) Failed testBitmaps().\n"));
		delete[] rgbQuadArr;
		rgbQuadArr = 0;
		return 1;
	}

	//Save image to a PNG format on disk.
	if (!savePNG(rgbQuadArr, params.width, params.height, "image.png")) {
		OutputDebugString(TEXT("(In ThreadProcRT()) Failed savePNG().\n"));
		delete[] rgbQuadArr;
		rgbQuadArr = 0;
		return 1;
	}

	delete[] rgbQuadArr;
	rgbQuadArr = 0;
	PostMessage(windowManager[0].hwnd, AppMsg_TFINISH, NULL, NULL);
	return 0;
}

//Heuristic. Not an inverse for the map spectrum -> rgb that is given in ColorModel.h.
void spectrumFromRGB(COLORREF rgbColor, std::vector<double>& spectrums) {
	double* spectrum = new double[STEPS];
	unsigned char stepLength = (MAX_WAVELENGTH - MIN_WAVELENGTH) / STEPS;
	unsigned short redStep = (650 - MIN_WAVELENGTH) / stepLength;
	unsigned short greenStep = (545 - MIN_WAVELENGTH) / stepLength;
	unsigned short blueStep = (450 - MIN_WAVELENGTH) / stepLength;

	memset(spectrum, 0, sizeof(double) * STEPS);
	spectrum[redStep] = 4.5 * (rgbColor & 0x000000FF) / 255.0;
	spectrum[greenStep] = 3.0 * ((rgbColor >> 8) & 0x0000FF) / 255.0;
	spectrum[blueStep] = 2.0 * ((rgbColor >> 16) & 0x000000FF) / 255.0;

	spectrums.insert(spectrums.end(), spectrum, spectrum + STEPS);

	delete[] spectrum;
}

void deleteSceneParams() {
	sceneParams.spectrums.clear();
	sceneParams.meshes.clear();
	sceneParams.faces.clear();
}

bool displayImage(unsigned char* rgbQuadArr, unsigned int cx, unsigned int cy) {
	RECT clientRect = {};
	HDC hdcMem = CreateCompatibleDC(NULL);
	HDC hdcClient = GetDC(windowManager[0].hwnd);
	HBITMAP hBitmap = CreateBitmap(cx, cy, 1, 32, rgbQuadArr);

	GetClientRect(windowManager[0].hwnd, &clientRect);
	if (!SelectObject(hdcMem, hBitmap)) {
		goto error;
		printf("failed selectobject()\n");
	}
	if (!BitBlt(hdcClient, (clientRect.right-clientRect.left)/8, 0, cx, cy, hdcMem, 0, 0, SRCCOPY)) {
		goto error;
		printf("failed bitblt()\n");
	}

	ReleaseDC(windowManager[0].hwnd, hdcClient);
	DeleteDC(hdcMem);
	DeleteObject(hBitmap);
	return true;
error:
	ReleaseDC(windowManager[0].hwnd, hdcClient);
	DeleteDC(hdcMem);
	DeleteObject(hBitmap);
	return false;
}