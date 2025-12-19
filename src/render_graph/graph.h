#pragma once

#include <vector>

namespace render_graph
{
    using pass_handle    = uint32_t;
    using resource_handle = uint32_t;

    // graph topology

    // one dimesion array to represent the read resource of each pass
    struct read_dependency
    {
        std::vector<resource_handle> read_list;
        std::vector<resource_handle> begins;
        std::vector<resource_handle> lengthes;
    };

    // one dimesion array to represent the write resource of each pass
    struct write_dependency
    {
        std::vector<resource_handle> write_list;
        std::vector<resource_handle> begins;
        std::vector<resource_handle> lengthes;
    };

    struct graph_topology
    {
        std::vector<pass_handle> passes;
    };

}; // namespace render_graph
