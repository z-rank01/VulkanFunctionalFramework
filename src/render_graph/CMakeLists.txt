# 查找 Vulkan 包
find_package(Vulkan REQUIRED)

add_library(render_graph STATIC
    graph.h
    graph.cpp
    resource.h
    resource_types.h
    backend.h
    vulkan_backend.h
    system.h
)

# 设置头文件包含目录
target_include_directories(render_graph
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..  # 引用其他模块目录下的头文件
)

# 链接 Vulkan
target_link_libraries(render_graph
    PUBLIC
        Vulkan::Vulkan
)