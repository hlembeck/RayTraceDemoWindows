#include "RayTraceDemoWindows.h"

#define AppMsg_TFINSH WM_APP

//Declarations of functions
void spectrumFromRGB(COLORREF rgbColor, std::vector<double> spectrums);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
DWORD WINAPI ThreadProcRT(LPVOID lpParameter);
void printImgParamPinhole(ImgParamPinhole& params);
void deleteSceneParams();

const TCHAR szDiagName[] = TEXT("Diagnostic Window");

static SceneParams sceneParams = {};

static CHOOSECOLOR chooseColorStruct = {};
static COLORREF acrCustClr[16];

//Vector of window handles
static std::vector<HWND> hWindows;
static HWND diagnosticWnd = 0;

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd) {
	static TCHAR szAppName[] = TEXT("Ray Tracing Demo");
	MSG msg = {};

	WNDCLASS baseWndClass = { CS_HREDRAW | CS_VREDRAW , WndProc, 0, 0, hInstance, NULL, LoadCursor(NULL, IDC_ARROW), (HBRUSH)GetStockObject(WHITE_BRUSH), NULL, szAppName };
	if (!RegisterClass(&baseWndClass)) {
		MessageBox(NULL, TEXT("Could not register the base window class."), szAppName, MB_ICONERROR);
		return -1;
	}

	WNDCLASS diagnosticWndClass = { CS_HREDRAW | CS_VREDRAW , DiagWndProc, 0, 0, hInstance, NULL, LoadCursor(NULL, IDC_ARROW), (HBRUSH)GetStockObject(WHITE_BRUSH), NULL, szDiagName };
	if (!RegisterClass(&diagnosticWndClass)) {
		MessageBox(NULL, TEXT("Could not register the diagnostics window class."), szDiagName, MB_ICONERROR);
		return -1;
	}

	hWindows.push_back(
		CreateWindow(szAppName, szAppName, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, LoadMenu(hInstance, MAKEINTRESOURCE(IDR_MENU1)), hInstance, NULL)
	);
	addParamWindows(hWindows, hInstance);

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
	//Used to transform meshes before they are added to faceVector above.
	static double* transformMatrix;

	static RTParams paramsRT = { {},sceneParams,NULL };
	static HANDLE hThread = 0;
	DWORD exitCode;

	switch (message) {
	case WM_CREATE:
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
					hWindows[0],
					NULL,
					0,
					acrCustClr,
					CC_FULLOPEN,
					NULL,
					NULL,
					NULL
				};
				if (ChooseColor(&chooseColorStruct)) {
					//Dialog box for matrix transform on the prism.
					addPrismHOST(sceneParams.faces, transformMatrix, sceneParams.spectrums.size());
					spectrumFromRGB(chooseColorStruct.rgbResult, sceneParams.spectrums);
					sceneParams.meshes.push_back(sceneParams.faces.size());
					if (!IsWindow(diagnosticWnd)) {
						diagnosticWnd = CreateWindow(szDiagName, szDiagName, WS_OVERLAPPEDWINDOW | WS_VSCROLL, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, GetModuleHandle(NULL), NULL);

						ShowWindow(diagnosticWnd, SW_SHOW);
					}
					SendMessage(diagnosticWnd, AppMsg_UpdateDiag, (WPARAM)&sceneParams, (LPARAM)&chooseColorStruct.rgbResult);
				}
				return 0;
			case 1:
				chooseColorStruct = {
					sizeof(CHOOSECOLOR),
					hWindows[0],
					NULL,
					0,
					acrCustClr,
					CC_FULLOPEN,
					NULL,
					NULL,
					NULL
				};
				if (ChooseColor(&chooseColorStruct)) {
					//Dialog box for matrix transform on the plate.
					addPlateHOST(sceneParams.faces, transformMatrix,sceneParams.spectrums.size());
					spectrumFromRGB(chooseColorStruct.rgbResult, sceneParams.spectrums);
					sceneParams.meshes.push_back(sceneParams.faces.size());
					if (!IsWindow(diagnosticWnd)) {
						diagnosticWnd = CreateWindow(szDiagName, szDiagName, WS_OVERLAPPEDWINDOW | WS_VSCROLL, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, GetModuleHandle(NULL), NULL);
						ShowWindow(diagnosticWnd, SW_SHOW);
					}
					SendMessage(diagnosticWnd, AppMsg_UpdateDiag, (WPARAM)&sceneParams, (LPARAM)&chooseColorStruct.rgbResult);
				}
				return 0;
			case 3:
				endPtr = 0;
				i = 0;
				memset(buf, 0, 576);
				SendMessage(hWindows[1], WM_GETTEXT, 32, (LPARAM)buf);
				params.pinhole.x = wcstod(buf, &endPtr);
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
				if (!hThread)
					hThread = CreateThread(NULL, 0, ThreadProcRT, &paramsRT, 0, NULL);
				return 0;
			}
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
		delete[] transformMatrix;
		deleteSceneParams();
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
	if (!getPinholeImage(*(RTParams*)lpParameter)) {
		OutputDebugString(TEXT("(In ThreadProcRT()) Failed getPinholeBitmap().\n"));
		return 1;
	}
	if (!savePNG(rgbQuadArr, params.width, params.height, "image.png")) {
		OutputDebugString(TEXT("(In ThreadProcRT()) Failed savePNG().\n"));
		return 1;
	}
	return 0;
}

//Heuristic. Not an inverse for the map spectrum -> rgb that is given in ColorModel.h.
void spectrumFromRGB(COLORREF rgbColor, std::vector<double> spectrums) {
	double* spectrum = new double[STEPS];
	unsigned char stepLength = (MAX_WAVELENGTH - MIN_WAVELENGTH) / STEPS;
	unsigned short redStep = (650 - MIN_WAVELENGTH) / stepLength;
	unsigned short greenStep = (545 - MIN_WAVELENGTH) / stepLength;
	unsigned short blueStep = (450 - MIN_WAVELENGTH) / stepLength;

	memset(spectrum, 0, sizeof(double) * STEPS);
	spectrum[redStep] = 3.0*(rgbColor & 0x000000FF)/255.0;
	spectrum[greenStep] = 2.0*(rgbColor & 0x0000FF00) / 255.0;
	spectrum[blueStep] = (rgbColor & 0x00FF0000) / 255.0;

	spectrums.insert(spectrums.end(), spectrum, spectrum + STEPS);

	delete[] spectrum;
}

void deleteSceneParams() {
	sceneParams.spectrums.clear();
	sceneParams.meshes.clear();
	sceneParams.faces.clear();
}