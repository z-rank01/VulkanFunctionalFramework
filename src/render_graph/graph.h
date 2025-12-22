#pragma once

#include <vector>

#include "resource.h"
#include "backend.h"

namespace render_graph
{
    using pass_handle       = uint32_t;
    using resource_handle   = uint32_t;
    
    // resource dependency

    // one dimesion array to represent the read resource of each pass
    struct read_dependency
    {
        std::vector<resource_handle> read_list;
        std::vector<resource_handle> begins;
        std::vector<resource_handle> lengthes;
        std::vector<resource_handle> generations;
    };

    // one dimesion array to represent the write resource of each pass
    struct write_dependency
    {
        std::vector<resource_handle> write_list;
        std::vector<resource_handle> begins;
        std::vector<resource_handle> lengthes;
        std::vector<resource_handle> generations;
    };

    // graph/pass context

    // context passed to the setup lambda
    struct pass_setup_context
    {
        resource_meta_table* meta_table;
        read_dependency* image_read_deps;
        write_dependency* image_write_deps;
        read_dependency* buffer_read_deps;
        write_dependency* buffer_write_deps;
        pass_handle current_pass;

        // create

        resource_handle create_image(const image_info& info) const { return meta_table->image_metas.add(info); }
        resource_handle create_buffer(const buffer_info& info) const { return meta_table->buffer_metas.add(info); }

        // read
        // at the stage of adding dependency of resource, no need to compute and designate generation of resource
        // generation will be computed through system inside compile function.

        void read_image(resource_handle resource) const
        {
            image_read_deps->read_list.push_back(resource);
            image_read_deps->lengthes[current_pass]++;
        }
        void read_buffer(resource_handle resource) const
        {
            buffer_read_deps->read_list.push_back(resource);
            buffer_read_deps->lengthes[current_pass]++;
        }

        // write
        // at the stage of adding dependency of resource, no need to compute and designate generation of resource
        // generation will be computed through system inside compile function.

        void write_image(resource_handle resource) const
        {
            image_write_deps->write_list.push_back(resource);
            image_write_deps->lengthes[current_pass]++;
        }
        void write_buffer(resource_handle resource) const
        {
            buffer_write_deps->write_list.push_back(resource);
            buffer_write_deps->lengthes[current_pass]++;
        }
    };

    // context passed to the execution lambda
    struct pass_execute_context
    {
        const backend* backend;
        // void* command_buffer; // Abstract command buffer
    };

    // graph topology

    using pass_execute_func = void (*)(pass_execute_context&);
    using pass_setup_func   = void (*)(pass_setup_context&);

    struct graph_topology
    {
        std::vector<pass_handle> passes;
        std::vector<pass_setup_func> setup_funcs;
        std::vector<pass_execute_func> execute_funcs;
    };

}; // namespace render_graph
