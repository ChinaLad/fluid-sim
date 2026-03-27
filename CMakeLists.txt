cmake_minimum_required(VERSION 4.2)
project(fluid_sim CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++")
set(CMAKE_CXX_FLAGS "-O3")

# 1. Find the packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

include_directories(src)

set_source_files_properties(gl.c PROPERTIES LANGUAGE CXX)
add_executable(fluid_sim
        src/simulation/particles.h
        src/main.cpp
        src/visualization/camera.h
        src/visualization/gl.c
        src/visualization/shader.h
        src/simulation/sim.cpp
        src/simulation/sim.h
        src/simulation/params.h
)

target_link_libraries(fluid_sim
        PUBLIC
        OpenGL::GL
        glfw
        rt dl m pthread
)