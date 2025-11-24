cmake_minimum_required(VERSION 3.20)

project(interface LANGUAGES CXX)

add_library(interface STATIC
        sdl_window.cpp
        sdl_window.h
        Window.h
        input.h
        camera.h
        camera.h
        simple_camera.cpp
        simple_camera.h
        simple_camera.h
)

target_link_libraries(interface
    PUBLIC
        SDL3::SDL3
        Vulkan::Vulkan
)

target_include_directories(interface
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
