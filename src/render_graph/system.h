#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "backend.h"
#include "graph.h"
#include "resource.h"

namespace render_graph
{
    // Context passed to the setup lambda
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

    // Context passed to the execution lambda
    struct pass_execute_context
    {
        const backend* backend;
        // void* command_buffer; // Abstract command buffer
    };

    using pass_execute_func = void (*)(pass_execute_context&);
    using pass_setup_func   = void (*)(pass_setup_context&);

    class render_graph_system
    {
    public:
        // resource related
        resource_meta_table meta_table;
        read_dependency image_read_deps;
        write_dependency image_write_deps;
        read_dependency buffer_read_deps;
        write_dependency buffer_write_deps;

        // pass related
        graph_topology graph;
        std::vector<pass_setup_func> setup_funcs;
        std::vector<pass_execute_func> execute_funcs;

        // backend
        backend* backend = nullptr;

        void set_backend(class backend* backend_ptr) { backend = backend_ptr; }

        // 1. Add Pass System
        // Separates resource definition (setup) from execution logic.
        template <typename SetupFn = pass_setup_func, typename ExecuteFn = pass_execute_func>
        pass_handle add_pass(SetupFn&& setup, ExecuteFn&& execute)
        {
            auto handle = static_cast<pass_handle>(graph.passes.size());
            graph.passes.push_back(handle);
            setup_funcs.push_back(std::forward<SetupFn>(setup));
            execute_funcs.push_back(std::forward<ExecuteFn>(execute));
            return handle;
        }

        // 2. Compile System (Culling & Allocation)
        void compile()
        {
            const auto pass_count = graph.passes.size();

            // Reset dependency storage (StepA will repopulate these)

            // Step A: Invoke Setup Functions
            // Invoke setup function to collect reousce usages so that we
            // can compute the topology of pass and execute succeeding phases.
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

                auto setup_func = setup_funcs[i];
                setup_func(setup_ctx);
            }

            // Step B: Culling
            // Analyze dependencies and mark passes as active/inactive
            // For now, we assume all passes are active.

            // Step C: Resource Allocation
            // 1. Filter out resources that are not used by active passes (if needed)
            // 2. Call backend to create physical resources
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
