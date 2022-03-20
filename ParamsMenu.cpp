#include "ParamsMenu.h"

void paintMenu(HDC hdc, RECT& rect) {
	int w = rect.right - rect.left;
	int h = rect.bottom - rect.top;

	SelectObject(hdc, GetStockObject(DC_BRUSH));
	SelectObject(hdc, GetStockObject(DC_PEN));
	SetDCBrushColor(hdc, RGB(150, 150, 150));
	SetDCPenColor(hdc, RGB(100, 100, 100));
	Rectangle(hdc, 0, 0, w / 8, h);
	SetBkColor(hdc, RGB(150, 150, 150));

	if (h < 510)
		return;

	TextOut(hdc, 0, 0, TEXT("Scene Menu:"), 11);

	TextOut(hdc, 0, 90, TEXT("Pinhole Coordinates:"), 20);
	TextOut(hdc, 0, 120, TEXT("X:"), 2);
	TextOut(hdc, 0, 150, TEXT("Y:"), 2);
	TextOut(hdc, 0, 180, TEXT("Z:"), 2);

	TextOut(hdc, 0, 210, TEXT("Image Dimensions:"), 17);
	TextOut(hdc, 0, 240, TEXT("Width:"), 6);
	TextOut(hdc, 0, 270, TEXT("Height:"), 7);

	TextOut(hdc, 0, 300, TEXT("Sensor Dimensions:"), 18);
	TextOut(hdc, 0, 330, TEXT("Width:"), 6);
	TextOut(hdc, 0, 360, TEXT("Height:"), 7);

	TextOut(hdc, 0, 390, TEXT("Ray Tracing Parameters:"), 23);
	TextOut(hdc, 0, 420, TEXT("sqrt(Rays/Pixel):"), 17);
	TextOut(hdc, 0, 450, TEXT("Reflections/Ray:"), 16);
}

void addParamWindows(HWND mainWindow, std::vector<HWND>& hWindows, HINSTANCE& hInstance) {
	//Scene Menu
	hWindows.push_back(CreateWindow(TEXT("BUTTON"), TEXT("Add Standard Prism"), WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 50, 30, 150, 20, mainWindow, (HMENU)0, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("BUTTON"), TEXT("Add Standard Plate"), WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 50, 60, 150, 20, mainWindow, (HMENU)1, hInstance, NULL));

	//Pinhole Parameter Menu
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 120, 100, 20, mainWindow, (HMENU)ID_EDIT, hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 150, 100, 20, mainWindow, (HMENU)(ID_EDIT + 1), hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 180, 100, 20, mainWindow, (HMENU)(ID_EDIT + 2), hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP | ES_NUMBER, 150, 240, 100, 20, mainWindow, (HMENU)(ID_EDIT + 3), hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP | ES_NUMBER, 150, 270, 100, 20, mainWindow, (HMENU)(ID_EDIT + 4), hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 330, 100, 20, mainWindow, (HMENU)(ID_EDIT + 5), hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP, 150, 360, 100, 20, mainWindow, (HMENU)(ID_EDIT + 6), hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP | ES_NUMBER, 150, 420, 100, 20, mainWindow, (HMENU)(ID_EDIT + 7), hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("edit"), NULL, WS_CHILD | WS_VISIBLE | WS_BORDER | WS_TABSTOP | ES_NUMBER, 150, 450, 100, 20, mainWindow, (HMENU)(ID_EDIT + 8), hInstance, NULL));
	hWindows.push_back(CreateWindow(TEXT("BUTTON"), TEXT("SUBMIT"), WS_TABSTOP | WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON, 50, 480, 100, 20, mainWindow, (HMENU)(ID_EDIT + 9), hInstance, NULL));
}