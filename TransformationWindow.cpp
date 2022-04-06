#include "TransformationWindow.h"

static HWND hMainWnd;

static std::vector<HWND> transformWindows;

static int state;

void updateState(HWND hWnd) {
	DestroyWindow(transformWindows[0]);
	DestroyWindow(transformWindows[1]);
	DestroyWindow(transformWindows[2]);
	DestroyWindow(transformWindows[3]);
	transformWindows.clear();
	switch (state) {
	case 0:
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			0,
			100,
			25,
			hWnd,
			(HMENU)transformationMenuID,
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			30,
			100,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 1),
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			60,
			100,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 2),
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("button"),
			TEXT("Submit"),
			WS_CHILD | WS_VISIBLE | WS_TABSTOP,
			50,
			90,
			100,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 7),
			NULL,
			NULL
		));
		return;
	case 1:
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			0,
			100,
			25,
			hWnd,
			(HMENU)transformationMenuID,
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			30,
			100,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 1),
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			60,
			100,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 2),
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("button"),
			TEXT("Submit"),
			WS_CHILD | WS_VISIBLE | WS_TABSTOP,
			50,
			90,
			100,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 7),
			NULL,
			NULL
		));
		return;
	case 2:
		//Vector
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			0,
			50,
			25,
			hWnd,
			(HMENU)transformationMenuID,
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			100,
			0,
			50,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 1),
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			150,
			0,
			50,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 2),
			NULL,
			NULL
		));

		//Angle
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			30,
			150,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 3),
			NULL,
			NULL
		));

		//Submit Button
		transformWindows.push_back(CreateWindow(
			TEXT("button"),
			TEXT("Submit"),
			WS_CHILD | WS_VISIBLE | WS_TABSTOP,
			50,
			60,
			150,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 7),
			NULL,
			NULL
		));
		break;
	}
}

Triple<double> getTriple(unsigned int start) {
	Triple<double> ret = {};
	TCHAR *buf = new TCHAR[64], *tBuf = buf;
	int bufIndex = 0;
	memset(buf, 0, sizeof(TCHAR) * 64);
	bufIndex += SendMessage(transformWindows[start], WM_GETTEXT, 16, (LPARAM)tBuf) + 1;
	ret.x = wcstod(tBuf, NULL);
	tBuf += bufIndex;
	bufIndex += SendMessage(transformWindows[start + 1], WM_GETTEXT, 16, (LPARAM)tBuf) + 1;
	ret.y = wcstod(tBuf, NULL);
	tBuf += bufIndex;
	SendMessage(transformWindows[start + 2], WM_GETTEXT, 16, (LPARAM)tBuf) + 1;
	ret.z = wcstod(tBuf, NULL);
	delete[] buf;
	return ret;
}

double getDouble(unsigned int index) {
	double ret;
	TCHAR* buf = new TCHAR[16];
	memset(buf, 0, sizeof(TCHAR) * 16);
	SendMessage(transformWindows[index], WM_GETTEXT, 16, (LPARAM)buf);
	ret = wcstod(buf, NULL);
	delete[] buf;
	return ret;
}

LRESULT CALLBACK TransformationWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	Triple<double> triple;
	double angle;
	double* transformMatrix;
	HDC hdc;
	PAINTSTRUCT ps;
	RECT clientRect;
	int clientHeight, clientWidth, bufIndex = 0;
	switch (message) {
	case WM_CREATE:
		state = -1;
		break;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		switch (state) {
		case 0:
			TextOut(hdc, 0, 0, L"X:", 2);
			TextOut(hdc, 0, 30, L"Y:", 2);
			TextOut(hdc, 0, 60, L"Z:", 2);
			break;
		case 1:
			TextOut(hdc, 0, 0, L"X:", 2);
			TextOut(hdc, 0, 30, L"Y:", 2);
			TextOut(hdc, 0, 60, L"Z:", 2);
			break;
		case 2:
			TextOut(hdc, 0, 0, L"Point:", 6);
			TextOut(hdc, 0, 30, L"Angle:", 6);
			break;
		}
		EndPaint(hWnd, &ps);
		return 0;
	case WM_SIZE:
		GetClientRect(hWnd, &clientRect);
		clientHeight = clientRect.bottom - clientRect.top;
		clientWidth = clientRect.right - clientRect.left;
		switch (state) {
		case -1:
			SetWindowPos(transformWindows[0], HWND_TOP, 0, 0, clientWidth, clientHeight / 4, 0);
			SetWindowPos(transformWindows[1], HWND_TOP, 0, clientHeight / 4, clientWidth, clientHeight / 4, 0);
			SetWindowPos(transformWindows[2], HWND_TOP, 0, clientHeight / 2, clientWidth, clientHeight / 4, 0);
			SetWindowPos(transformWindows[3], HWND_TOP, 0, 3 * clientHeight / 4, clientWidth, clientHeight / 4, 0);
		}
		return 0;
	case WM_COMMAND:
		switch (HIWORD(wParam)) {
		case BN_CLICKED:
			switch (LOWORD(wParam)) {
			//Translate
			case transformationMenuID:
				state = 0;
				updateState(hWnd);
				return 0;
			//Scale
			case transformationMenuID + 1:
				state = 1;
				updateState(hWnd);
				return 0;
			//Rotate
			case transformationMenuID + 2:
				state = 2;
				updateState(hWnd);
				return 0;
			//Matrix
			case transformationMenuID + 3:
				PostMessage(hMainWnd, AppMsg_TransformationWindow, 3, 0);
				return 0;
			//Button press
			case transformationMenuID + 7:
				switch (state) {
				case 0:
					transformMatrix = new double[16];
					memset(transformMatrix, 0, sizeof(double) * 16);
					triple = getTriple(0);
					transformMatrix[0] = 1;
					transformMatrix[3] = triple.x;
					transformMatrix[5] = 1;
					transformMatrix[7] = triple.y;
					transformMatrix[10] = 1;
					transformMatrix[11] = triple.z;
					transformMatrix[15] = 1;


					PostMessage(hMainWnd, AppMsg_TransformationWindow + 1, (WPARAM)transformMatrix, NULL);

					break;
				case 1:
					transformMatrix = new double[16];
					memset(transformMatrix, 0, sizeof(double) * 16);
					triple = getTriple(0);
					transformMatrix[0] = triple.x;
					transformMatrix[5] = triple.y;
					transformMatrix[10] = triple.z;
					transformMatrix[15] = 1;

					PostMessage(hMainWnd, AppMsg_TransformationWindow + 1, (WPARAM)transformMatrix, NULL);

					break;
				case 2:
					transformMatrix = getRotateTransformHOST(getTriple(0), getDouble(3));

					PostMessage(hMainWnd, AppMsg_TransformationWindow + 1, (WPARAM)transformMatrix, NULL);

					break;
				}
				return 0;
			}
		}
	}

	return DefWindowProc(hWnd,message,wParam,lParam);
}

windowInfo createTransformationWindow(HWND hWndMain, HINSTANCE hInstacnce) {
	hMainWnd = hWndMain;
	windowInfo ret = {
		CreateWindow(
			szTransformationWndName,
			szTransformationWndName,
			WS_EX_CONTROLPARENT | WS_OVERLAPPEDWINDOW,
			100,
			100,
			400,
			400,
			NULL,
			NULL,
			NULL,
			NULL
		),
		transformWindows};
	//Button to display translate menu for the mesh
	transformWindows.push_back(CreateWindow(
		TEXT("BUTTON"),
		TEXT("Translate"),
		WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		0,
		0,
		75,
		25,
		ret.hwnd,
		(HMENU)transformationMenuID,
		NULL,
		NULL
	));

	//Button to display scale menu for the mesh
	transformWindows.push_back(CreateWindow(
		TEXT("BUTTON"),
		TEXT("Scale"),
		WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		0,
		30,
		75,
		25,
		ret.hwnd,
		(HMENU)(transformationMenuID + 1),
		NULL,
		NULL
	));

	//Button to display rotate menu for the mesh
	transformWindows.push_back(CreateWindow(
		TEXT("BUTTON"),
		TEXT("Rotate"),
		WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		0,
		60,
		75,
		25,
		ret.hwnd,
		(HMENU)(transformationMenuID  + 2),
		NULL,
		NULL
	));

	//Button to display matrix menu for the mesh
	transformWindows.push_back(CreateWindow(
		TEXT("BUTTON"),
		TEXT("Matrix"),
		WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
		0,
		90,
		75,
		25,
		ret.hwnd,
		(HMENU)(transformationMenuID + 3),
		NULL,
		NULL
	));

	ShowWindow(ret.hwnd, SW_SHOW);
	UpdateWindow(ret.hwnd);

	return ret;
}