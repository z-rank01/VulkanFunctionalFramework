# Unit-test sources for render_graph.
# These tests are compiled into the main executable (via OBJECT library objects)
# so they are always built even if the test functions are not referenced.

add_library(render_graph_unit_tests OBJECT
    deferred_rendering_compile_test.cpp
    resource_producer_map_compile_test.cpp
    resource_generation_compile_test.cpp
    culling_compile_test.cpp
)

target_link_libraries(render_graph_unit_tests
    PRIVATE
        render_graph
)

# Ensure includes like "render_graph/system.h" resolve.
# render_graph already exports ${CMAKE_CURRENT_SOURCE_DIR}/.. (i.e., src/),
# but we keep this explicit for the unit-test target.
target_include_directories(render_graph_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
