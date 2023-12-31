cmake_minimum_required(VERSION 3.12)

project(videoStabilization VERSION 1.0.0)

# Set OpenCV
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

# Compile options
set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin")

FILE(GLOB ProjectSources ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.c ${PROJECT_SOURCE_DIR}/src/**/*.cpp ${PROJECT_SOURCE_DIR}/src/**/*.c)

# Thirdparty libraries
add_subdirectory(thirdparty/freeglut-3.2.1)

# Project files
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(thirdparty/freeglut-3.2.1/include)

# Compile
add_executable(videoStabilization ${ProjectSources})
target_link_libraries(videoStabilization freeglut_static ${OPENGL_LIBRARIES} ${OpenCV_LIBS})