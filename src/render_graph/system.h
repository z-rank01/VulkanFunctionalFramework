#pragma once

#include <limits>
#include <vector>

#include "backend.h"
#include "graph.h"
#include "resource.h"

namespace render_graph
{
    class render_graph_system
    {
    public:
        // resource related
        resource_meta_table meta_table;
        read_dependency image_read_deps;
        write_dependency image_write_deps;
        read_dependency buffer_read_deps;
        write_dependency buffer_write_deps;
        resource_producer_lookup_table producer_lookup_table;

        // pass related
        graph_topology graph;
        
        // backend related
        backend* backend = nullptr;

        void set_backend(class backend* backend_ptr) { backend = backend_ptr; }

        // 1. Add Pass System
        // Separates resource definition (setup) from execution logic.
        template <typename SetupFn = pass_setup_func, typename ExecuteFn = pass_execute_func>
        pass_handle add_pass(SetupFn&& setup, ExecuteFn&& execute)
        {
            auto handle = static_cast<pass_handle>(graph.passes.size());
            graph.passes.push_back(handle);
            graph.setup_funcs.push_back(std::forward<SetupFn>(setup));
            graph.execute_funcs.push_back(std::forward<ExecuteFn>(execute));
            return handle;
        }

        // 2. Compile System (Culling & Allocation)
        void compile()
        {
            const auto pass_count = graph.passes.size();

            // Step A: Invoke Setup Functions
            // Invoke setup function to collect reousce usages so that we
            // can compute the topology of pass and execute succeeding phases.
            
            // Reset dependency storage
            image_read_deps.read_list.clear();
            image_read_deps.begins.assign(pass_count, 0);
            image_read_deps.lengthes.assign(pass_count, 0);
            image_write_deps.write_list.clear();
            image_write_deps.begins.assign(pass_count, 0);
            image_write_deps.lengthes.assign(pass_count, 0);
            buffer_read_deps.read_list.clear();
            buffer_read_deps.begins.assign(pass_count, 0);
            buffer_read_deps.lengthes.assign(pass_count, 0);
            buffer_write_deps.write_list.clear();
            buffer_write_deps.begins.assign(pass_count, 0);
            buffer_write_deps.lengthes.assign(pass_count, 0);
            pass_setup_context setup_ctx{.meta_table        = &meta_table,
                                         .image_read_deps   = &image_read_deps,
                                         .image_write_deps  = &image_write_deps,
                                         .buffer_read_deps  = &buffer_read_deps,
                                         .buffer_write_deps = &buffer_write_deps,
                                         .current_pass      = 0};
            for (size_t i = 0; i < pass_count; i++)
            {
                setup_ctx.current_pass = graph.passes[i];

                // Mark begin offsets for this pass (SoA range encoding)
                image_read_deps.begins[setup_ctx.current_pass]   = static_cast<pass_handle>(image_read_deps.read_list.size());
                image_write_deps.begins[setup_ctx.current_pass]  = static_cast<pass_handle>(image_write_deps.write_list.size());
                buffer_read_deps.begins[setup_ctx.current_pass]  = static_cast<pass_handle>(buffer_read_deps.read_list.size());
                buffer_write_deps.begins[setup_ctx.current_pass] = static_cast<pass_handle>(buffer_write_deps.write_list.size());

                auto setup_func = graph.setup_funcs[i];
                setup_func(setup_ctx);
            }

            // Step B: Build resource-producer map
            // Map each resource to the pass that writes to it.
            // Accelerate lookups during culling and execution.
            
            // TODO: create and take account of resource version
            auto image_resource_count = meta_table.image_metas.names.size();
            auto buffer_resource_count = meta_table.buffer_metas.names.size();
            producer_lookup_table.img_proc_map.assign(image_resource_count, std::numeric_limits<pass_handle>::max());
            producer_lookup_table.buf_proc_map.assign(buffer_resource_count, std::numeric_limits<pass_handle>::max());
            for(size_t i = 0; i < pass_count; i++)
            {
                auto current_pass = graph.passes[i];
                auto begin = image_write_deps.begins[current_pass];
                auto length = image_write_deps.lengthes[current_pass];
                for(auto j = begin; j < begin + length; j++)
                {
                    auto image = image_write_deps.write_list[j];
                    producer_lookup_table.img_proc_map[image] = current_pass;
                }
            }
            for(size_t i = 0; i < pass_count; i++)
            {
                auto current_pass = graph.passes[i];
                auto begin = buffer_write_deps.begins[current_pass];
                auto length = buffer_write_deps.lengthes[current_pass];
                for(auto j = begin; j < begin + length; j++)
                {
                    auto buffer = buffer_write_deps.write_list[j];
                    producer_lookup_table.buf_proc_map[buffer] = current_pass;
                }
            }

            // Step C: Culling
            // Analyze dependencies and mark passes as active/inactive

            // Step D: Resource Allocation
            // 1. Filter out resources that are not used by active passes (if needed)
            // 2. Call backend to create physical resources

            // Step E: 
        }

        // 3. Execution System
        void execute() { }

        void clear()
        {
            meta_table.clear();
            if (backend != nullptr)
            {
                backend->destroy_resources();
            }
        }
    };

} // namespace render_graph
