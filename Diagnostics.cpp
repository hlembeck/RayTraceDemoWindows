#include "Diagnostics.h"

static HFONT defFont = {};

static std::vector<COLORREF> sampleColors;

void handlePaintDiag(SceneParams& sceneParams, HDC hdc, RECT& rect, unsigned int scrollPosition);

LRESULT CALLBACK DiagWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
	HDC hdc;
	PAINTSTRUCT ps;
	RECT rect;
	static SceneParams* sceneParams = 0;
	static unsigned int scrollPosition = 0;
	switch (message) {
	case WM_CREATE:
		//sampleColors.push_back(*(COLORREF*)(((CREATESTRUCT*)lParam)->lpCreateParams));
		return 0;
	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		GetClientRect(hWnd, &rect);
		if (sceneParams)
			handlePaintDiag(*sceneParams, hdc, rect, scrollPosition);
		EndPaint(hWnd, &ps);
		return 0;
	case AppMsg_UpdateDiag:
		sceneParams = (SceneParams*)wParam;
		sampleColors.push_back(*((COLORREF*)lParam));
		GetClientRect(hWnd, &rect);
		InvalidateRect(hWnd, &rect, FALSE);
		UpdateWindow(hWnd);
		return 0;
	case WM_VSCROLL:
		switch (LOWORD(wParam)) {
		case SB_THUMBTRACK:
			
		case SB_THUMBPOSITION:
			scrollPosition = HIWORD(wParam);
			SetScrollPos(hWnd, SB_VERT, scrollPosition, TRUE);
			GetClientRect(hWnd, &rect);
			InvalidateRect(hWnd, &rect, TRUE);
			UpdateWindow(hWnd);
		}
		return 0;
	}
	return DefWindowProc(hWnd, message, wParam, lParam);
}

void handlePaintDiag(SceneParams& sceneParams, HDC hdc, RECT& rect, unsigned int scrollPosition) {
	TEXTMETRIC textMetrics = {};
	GetTextMetrics(hdc, &textMetrics);
	const unsigned int bufLen = (rect.right - rect.left) / textMetrics.tmMaxCharWidth;
	unsigned int bufUsed;
	RECT colorSampleRect;
	TCHAR* buf = new TCHAR[bufLen];
	HFONT defFont = {}, currFont = CreateFont(2 * textMetrics.tmHeight, 2 * textMetrics.tmAveCharWidth, 0, 0, textMetrics.tmWeight, FALSE, FALSE, FALSE, textMetrics.tmCharSet, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, textMetrics.tmPitchAndFamily, NULL);

	if (bufLen < 30 || sceneParams.meshes.size() == 0) {
		OutputDebugString(TEXT("SIZE 0 or buflen<100.\n"));
		delete[] buf;
		return;
	}

	defFont = (HFONT)SelectObject(hdc, currFont);
	SelectObject(hdc, defFont);

	TextOut(hdc, 0, -scrollPosition, buf, swprintf(buf, bufLen, L"Faces of Mesh 0:"));
	for (unsigned int i = 0; i < sceneParams.meshes[0]; i++) {
		TextOut(hdc, textMetrics.tmAveCharWidth, (6 * i + 1 - scrollPosition)*textMetrics.tmHeight, L"Face:", 5);

		//Rectangle to show color of the face.
		colorSampleRect = {textMetrics.tmAveCharWidth + 5*textMetrics.tmMaxCharWidth, (LONG)(6 * i + 1 - scrollPosition) * textMetrics.tmHeight, textMetrics.tmAveCharWidth + 5 * textMetrics.tmMaxCharWidth + 10, (LONG)(6 * i + 2 - scrollPosition) * textMetrics.tmHeight };

		FillRect(hdc, &colorSampleRect, CreateSolidBrush(sampleColors[0]));

		TextOut(hdc, 2 * textMetrics.tmAveCharWidth, (6 * i + 2 - scrollPosition) * textMetrics.tmHeight, buf, swprintf(buf, bufLen, L"%f %f %f", sceneParams.faces[i].p1.x, sceneParams.faces[i].p1.y, sceneParams.faces[i].p1.z));

		TextOut(hdc, 2 * textMetrics.tmAveCharWidth, (6 * i + 3 - scrollPosition) * textMetrics.tmHeight, buf, swprintf(buf, bufLen, L"%f %f %f", sceneParams.faces[i].p2.x, sceneParams.faces[i].p2.y, sceneParams.faces[i].p2.z));

		TextOut(hdc, 2 * textMetrics.tmAveCharWidth, (6 * i + 4 - scrollPosition) * textMetrics.tmHeight, buf, swprintf(buf, bufLen, L"%f %f %f", sceneParams.faces[i].p3.x, sceneParams.faces[i].p3.y, sceneParams.faces[i].p3.z));

		TextOut(hdc, textMetrics.tmAveCharWidth, (6 * i + 5 - scrollPosition) * (textMetrics.tmHeight), L"Normal:", 7);

		TextOut(hdc, 2 * textMetrics.tmAveCharWidth, (6 * (i + 1) - scrollPosition) * (textMetrics.tmHeight), buf, swprintf(buf, bufLen, L"%f %f %f", sceneParams.faces[i].n.x, sceneParams.faces[i].n.y, sceneParams.faces[i].n.z));
	}
	
	for (unsigned int i = 1; i < sceneParams.meshes.size(); i++) {
		TextOut(hdc, 0, (i + 6*sceneParams.meshes[i-1] - scrollPosition)*textMetrics.tmHeight, buf, swprintf(buf, bufLen, L"Faces of Mesh %d:",i));

		for (unsigned int j = sceneParams.meshes[i - 1]; j < sceneParams.meshes[i]; j++) {
			TextOut(hdc, textMetrics.tmAveCharWidth, (i + 6 * sceneParams.meshes[i - 1] - scrollPosition) * textMetrics.tmHeight + (6 * (j - sceneParams.meshes[i-1]) + 1) * textMetrics.tmHeight, L"Face:", 5);

			//Rectangle to show color of the face.
			colorSampleRect = {
				textMetrics.tmAveCharWidth + 5 * textMetrics.tmMaxCharWidth,
				(LONG)((i + 6 * sceneParams.meshes[i - 1] - scrollPosition) * textMetrics.tmHeight + (6 * (j - sceneParams.meshes[i - 1]) + 1) * textMetrics.tmHeight),
				textMetrics.tmAveCharWidth + 5 * textMetrics.tmMaxCharWidth + 10,
				(LONG)((1 + i + 6 * sceneParams.meshes[i - 1] - scrollPosition) * textMetrics.tmHeight + (6 * (j - sceneParams.meshes[i - 1]) + 1) * textMetrics.tmHeight)
			};

			FillRect(hdc, &colorSampleRect, CreateSolidBrush(sampleColors[i]));

			TextOut(hdc, 2 * textMetrics.tmAveCharWidth, (i + 6 * sceneParams.meshes[i - 1] - scrollPosition) * textMetrics.tmHeight + (6 * (j - sceneParams.meshes[i - 1]) + 2) * textMetrics.tmHeight, buf, swprintf(buf, bufLen, L"%f %f %f", sceneParams.faces[j].p1.x, sceneParams.faces[j].p1.y, sceneParams.faces[j].p1.z));

			TextOut(hdc, 2  * textMetrics.tmAveCharWidth, (i + 6 * sceneParams.meshes[i - 1] - scrollPosition) * textMetrics.tmHeight + (6 * (j - sceneParams.meshes[i - 1]) + 3) * textMetrics.tmHeight, buf, swprintf(buf, bufLen, L"%f %f %f", sceneParams.faces[j].p2.x, sceneParams.faces[j].p2.y, sceneParams.faces[j].p2.z));

			TextOut(hdc, 2 * textMetrics.tmAveCharWidth, (i + 6 * sceneParams.meshes[i - 1] - scrollPosition) * textMetrics.tmHeight + (6 * (j - sceneParams.meshes[i - 1]) + 4) * textMetrics.tmHeight, buf, swprintf(buf, bufLen, L"%f %f %f", sceneParams.faces[j].p3.x, sceneParams.faces[j].p3.y, sceneParams.faces[j].p3.z));

			TextOut(hdc, 2 * textMetrics.tmAveCharWidth, (i + 6 * sceneParams.meshes[i - 1] - scrollPosition) * textMetrics.tmHeight + (6 * (j - sceneParams.meshes[i - 1]) + 5) * textMetrics.tmHeight, L"Normal:", 7);

			TextOut(hdc, 2 * textMetrics.tmAveCharWidth, (i + 6 * sceneParams.meshes[i - 1] - scrollPosition) * textMetrics.tmHeight + 6 * (j - sceneParams.meshes[i - 1] + 1) * textMetrics.tmHeight, buf, swprintf(buf, bufLen, L"%f %f %f", sceneParams.faces[j].n.x, sceneParams.faces[j].n.y, sceneParams.faces[j].n.z));
		}
	}

	DeleteObject(currFont);
	delete[] buf;
}