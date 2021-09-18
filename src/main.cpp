#include "process.hpp"

bool cuda_init(int &deviceNum)
{

    int number = cv::cuda::getCudaEnabledDeviceCount();
    int choose;

    if (number == 0)
    {
        std::cout << "CUDA is not available" << std::endl;
        return false;
    }

    if (number == -1)
    {
        std::cout << "CUDA driver is not installed or version is not supported" << std::endl;
        return false;
    }

    for (int i = 0; i < number; i++)
    {
        cv::cuda::printShortCudaDeviceInfo(i);
        std::cout << std::endl;
    }

    if (number == 1)
    {
        deviceNum = 0;
        return true;
    }

    std::cout << "Please select the device" << std::endl;
select:;
    std::cin >> choose;

    if (choose < 0 || choose >= number)
    {
        std::cout << "Please enter valid device number" << std::endl;
        goto select;
    }

    deviceNum = choose;

    return true;
}

int main(int argc, char *argv[])
{
    // Check number of inputs
    if (argc % 2 != 1)
    {
        printf("Usage ./videoStabilization {Input} {Output}\n\n");
        printf("\tPossible input values:\n");
        printf("\t\t Path to a video file\n");
        printf("\t\t Camera num\n\n");
        return EXIT_FAILURE;
    }
    char vid1[] = "C:\\Users\\egece\\Videos\\4K Road traffic video for object detection and tracking.mp4";
    char vid2[] = "0";

    // Init CUDA device
    int deviceNum = -1;
    if (!cuda_init(deviceNum))
        return EXIT_FAILURE;

    // Init GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    mainWindow = createMainWindow("video");
    createSubWindow(mainWindow, 2, 2);

    // Start threads
    std::vector<std::thread> vTh;
    std::thread expThread(mainProcess, vid1, 0, deviceNum, true);
    vTh.push_back(std::move(expThread));
    for (size_t idx = 1; idx < argc; ++idx)
    {
        std::thread th(mainProcess, argv[idx], idx - 1, deviceNum, true);
        vTh.push_back(std::move(th));
    }

    // Join
    glutMainLoop();
    for (auto &entry : vTh)
        entry.join();

    destroyWindow(mainWindow);

    return 0;
}