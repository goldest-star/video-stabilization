#include <vector>

#include <opencv2/opencv.hpp>

#include <Windows.h>
#include <gl/GL.h>
#include <gl/glut.h>

#define FPS 60

extern GLint mainWindow;
extern std::vector<GLint> subWindows;
extern std::vector<cv::Mat> renderBuffer;

void createSubWindow(GLint mainWin, int maxX, int maxY);
GLint createMainWindow(const char *winName);
void updateWindow(GLint win, cv::Mat img);
void destroyWindow(GLint win);