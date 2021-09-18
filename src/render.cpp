#include "render.hpp"

bool loopFlag;
GLint mainWindow;
std::vector<GLint> subWindows;
std::vector<GLuint> textureBuffer;
std::vector<cv::Mat> renderBuffer;

void keyboardFunc(unsigned char key, int x, int y)
{
    loopFlag = false;
#ifdef _WIN32
    Sleep(1000);
#else
    usleep(1000);
#endif
    if (key == 81 || key == 113)
        glutLeaveMainLoop();
}

void timerFunc(int val)
{
    if (loopFlag)
    {
        glutSetWindow(mainWindow);
        glutPostRedisplay();
        glutTimerFunc(1000 / FPS, timerFunc, 0);
    }
}

void refreshMainFunc()
{
    for (size_t idx = 0; idx < subWindows.size(); ++idx)
    {
        // Choose window
        glutSetWindow(subWindows[idx]);
        glutPostRedisplay();
    }
}

void displayFunc()
{
    GLint win = glutGetWindow();
    auto it = std::find(subWindows.begin(), subWindows.end(), win);
    if (it != subWindows.end())
    {
        int indx = it - subWindows.begin();

        if (!renderBuffer[indx].empty())
        {
            // Prepare texture
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, textureBuffer[indx]);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, renderBuffer[indx].cols, renderBuffer[indx].rows, 0, GL_RGB, GL_UNSIGNED_BYTE, renderBuffer[indx].data);

            glBegin(GL_QUADS);
            glTexCoord2d(0.0, 0.0);
            glVertex2d(-1.0, +1.0);
            glTexCoord2d(1.0, 0.0);
            glVertex2d(+1.0, +1.0);
            glTexCoord2d(1.0, 1.0);
            glVertex2d(+1.0, -1.0);
            glTexCoord2d(0.0, 1.0);
            glVertex2d(-1.0, -1.0);
            glEnd();

            glDisable(GL_TEXTURE_2D);
            glutSwapBuffers();
        }
    }
}

void createSubWindow(GLint mainWin, int maxX, int maxY)
{
    int width = glutGet(GLUT_SCREEN_WIDTH);
    int height = glutGet(GLUT_SCREEN_HEIGHT);

    subWindows.clear();
    renderBuffer.clear();
    for (int idx = 0; idx < maxX * maxY; ++idx)
    {
        subWindows.push_back(glutCreateSubWindow(mainWin, (idx % maxX) * width / maxX, (idx % maxY) * height / maxY, width / maxX, height / maxY));
        glutSetWindow(subWindows[idx]);
        glutDisplayFunc(displayFunc);

        GLuint tex;
        glGenTextures(1, &tex);

        renderBuffer.push_back(cv::Mat());
        textureBuffer.push_back(tex);
    }
}

GLint createMainWindow(const char *winName)
{
    int width = glutGet(GLUT_SCREEN_WIDTH);
    int height = glutGet(GLUT_SCREEN_HEIGHT);

    glutInitWindowSize(width, height);
    glutInitWindowPosition(0, 0);
    GLint retval = glutCreateWindow(winName);
    glutDisplayFunc(refreshMainFunc);
    glutKeyboardFunc(keyboardFunc);
    glutTimerFunc(1000 / FPS, timerFunc, 0);

    return retval;
}

void updateWindow(GLint win, cv::Mat img)
{
    auto it = std::find(subWindows.begin(), subWindows.end(), win);
    if (it != subWindows.end())
    {
        int indx = it - subWindows.begin();
        img.copyTo(renderBuffer[indx]);
    }
}

void destroyWindow(GLint win)
{
    for (const auto &entry : subWindows)
        glutDestroyWindow(entry);
    glutDestroyWindow(win);
}
