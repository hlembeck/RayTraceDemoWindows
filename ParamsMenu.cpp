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

	if (h < 390)
		return;

	TextOut(hdc, 0, 0, TEXT("Pinhole Coordinates:"), 20);
	TextOut(hdc, 0, 30, TEXT("X:"), 2);
	TextOut(hdc, 0, 60, TEXT("Y:"), 2);
	TextOut(hdc, 0, 90, TEXT("Z:"), 2);

	TextOut(hdc, 0, 120, TEXT("Image Dimensions:"), 17);
	TextOut(hdc, 0, 150, TEXT("Width:"), 6);
	TextOut(hdc, 0, 180, TEXT("Height:"), 7);

	TextOut(hdc, 0, 210, TEXT("Sensor Dimensions:"), 18);
	TextOut(hdc, 0, 240, TEXT("Width:"), 6);
	TextOut(hdc, 0, 270, TEXT("Height:"), 7);

	TextOut(hdc, 0, 300, TEXT("Ray Tracing Parameters:"), 23);
	TextOut(hdc, 0, 330, TEXT("sqrt(Rays/Pixel):"), 17);
	TextOut(hdc, 0, 360, TEXT("Reflections/Ray:"), 16);
}