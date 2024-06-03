cmake_minimum_required(VERSION 3.5)
project(OpenGL LANGUAGES CXX)

cmake_policy(SET CMP0072 NEW)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

set(OpenGL_GL_PREFERENCE "GLVND")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL GLEW::GLEW glfw)