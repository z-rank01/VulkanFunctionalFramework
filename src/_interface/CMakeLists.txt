cmake_minimum_required(VERSION 3.20)

project(platform LANGUAGES CXX)

add_library(platform STATIC
    sdl_window.cpp
    sdl_window.h
    window.h
    input.h
)

target_link_libraries(platform
    PUBLIC
        SDL3::SDL3
        Vulkan::Vulkan
)

target_include_directories(platform
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
