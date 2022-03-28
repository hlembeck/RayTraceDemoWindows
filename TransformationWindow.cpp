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
		//Point
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

		//Vector
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			30,
			50,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 3),
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			100,
			30,
			50,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 4),
			NULL,
			NULL
		));
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			150,
			30,
			50,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 5),
			NULL,
			NULL
		));

		//Angle
		transformWindows.push_back(CreateWindow(
			TEXT("edit"),
			NULL,
			WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP,
			50,
			60,
			150,
			25,
			hWnd,
			(HMENU)(transformationMenuID + 6),
			NULL,
			NULL
		));

		//Submit Button
		transformWindows.push_back(CreateWindow(
			TEXT("button"),
			TEXT("Submit"),
			WS_CHILD | WS_VISIBLE | WS_TABSTOP,
			50,
			90,
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

LRESULT CALLBACK TransformationWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	HDC hdc;
	PAINTSTRUCT ps;
	RECT clientRect;
	int clientHeight, clientWidth;
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
			TextOut(hdc, 0, 30, L"Vector:", 7);
			TextOut(hdc, 0, 60, L"Angle:", 6);
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
			case transformationMenuID + 7:
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