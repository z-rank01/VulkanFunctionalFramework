#pragma once

#include <limits>
#include <queue>
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
        output_table output_table;

        // pass related
        graph_topology graph;
        std::vector<bool> active_pass_flags;

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

        // 2. Compile System

        void compile()
        {
            const auto pass_count = graph.passes.size();

            // Reset dependency storage
            image_read_deps.read_list.clear();
            image_read_deps.begins.assign(pass_count, 0);
            image_read_deps.lengthes.assign(pass_count, 0);
            image_read_deps.generations.clear();
            image_write_deps.write_list.clear();
            image_write_deps.begins.assign(pass_count, 0);
            image_write_deps.lengthes.assign(pass_count, 0);
            image_write_deps.generations.clear();
            buffer_read_deps.read_list.clear();
            buffer_read_deps.begins.assign(pass_count, 0);
            buffer_read_deps.lengthes.assign(pass_count, 0);
            buffer_read_deps.generations.clear();
            buffer_write_deps.write_list.clear();
            buffer_write_deps.begins.assign(pass_count, 0);
            buffer_write_deps.lengthes.assign(pass_count, 0);
            buffer_write_deps.generations.clear();
            output_table.image_outputs.clear();
            output_table.buffer_outputs.clear();

            // Step A: Invoke Setup Functions
            // Invoke setup function to collect resource usages so that we
            // can compute the topology of pass and execute succeeding phases.
            // - Read: graph.passes, graph.setup_funcs
            // - Write: meta_table, image_read_deps, image_write_deps, buffer_read_deps, buffer_write_deps

            pass_setup_context setup_ctx{.meta_table        = &meta_table,
                                         .image_read_deps   = &image_read_deps,
                                         .image_write_deps  = &image_write_deps,
                                         .buffer_read_deps  = &buffer_read_deps,
                                         .buffer_write_deps = &buffer_write_deps,
                                         .output_table      = &output_table,
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
            // Accelerate lookups during culling and dag construction.
            // - Read: graph.passes, image_write_deps, buffer_write_deps
            // - Write: producer_lookup_table

            const auto image_count  = meta_table.image_metas.names.size();
            const auto buffer_count = meta_table.buffer_metas.names.size();
            producer_lookup_table.img_proc_map.assign(image_count, std::numeric_limits<pass_handle>::max());
            producer_lookup_table.buf_proc_map.assign(buffer_count, std::numeric_limits<pass_handle>::max());
            for (size_t i = 0; i < pass_count; i++)
            {
                auto current_pass = graph.passes[i];
                auto begin        = image_write_deps.begins[current_pass];
                auto length       = image_write_deps.lengthes[current_pass];
                for (auto j = begin; j < begin + length; j++)
                {
                    auto image                                = image_write_deps.write_list[j];
                    producer_lookup_table.img_proc_map[image] = current_pass;
                }
            }
            for (size_t i = 0; i < pass_count; i++)
            {
                auto current_pass = graph.passes[i];
                auto begin        = buffer_write_deps.begins[current_pass];
                auto length       = buffer_write_deps.lengthes[current_pass];
                for (auto j = begin; j < begin + length; j++)
                {
                    auto buffer                                = buffer_write_deps.write_list[j];
                    producer_lookup_table.buf_proc_map[buffer] = current_pass;
                }
            }

            // Step C: Compute Resource Generation
            // compute generation list inside image and buffer dependency structure,
            // prepare for graph culling and dag construction.
            // - Read: graph.passes, image_read_deps, image_write_deps, buffer_read_deps, buffer_write_deps
            // - Write: image_read_deps.generations, image_write_deps.generations, buffer_read_deps.generations, buffer_write_deps.generations

            // Prepare generation arrays.
            const auto image_read_count   = image_read_deps.read_list.size();
            const auto image_write_count  = image_write_deps.write_list.size();
            const auto buffer_read_count  = buffer_read_deps.read_list.size();
            const auto buffer_write_count = buffer_write_deps.write_list.size();
            image_read_deps.generations.assign(image_read_count, 0);
            image_write_deps.generations.assign(image_write_count, 0);
            buffer_read_deps.generations.assign(buffer_read_count, 0);
            buffer_write_deps.generations.assign(buffer_write_count, 0);

            // We store the "next generation" to be assigned on a write.
            // - On write: gen = next_gen; next_gen++
            // - On read: gen = (next_gen == 0) ? 0 : (next_gen - 1)
            std::vector<resource_handle> image_next_gen(image_count, 0);
            std::vector<resource_handle> buffer_next_gen(buffer_count, 0);
            for (size_t i = 0; i < pass_count; i++)
            {
                auto current_pass = graph.passes[i];

                // compute image generation that current pass reads.
                auto read_begin  = image_read_deps.begins[current_pass];
                auto read_length = image_read_deps.lengthes[current_pass];
                for (auto j = read_begin; j < read_begin + read_length; j++)
                {
                    auto image                     = image_read_deps.read_list[j];
                    auto next_gen                  = image_next_gen[image];
                    image_read_deps.generations[j] = (next_gen == 0) ? 0 : (next_gen - 1);
                }

                // compute image generation that current pass writes.
                auto write_begin  = image_write_deps.begins[current_pass];
                auto write_length = image_write_deps.lengthes[current_pass];
                for (auto j = write_begin; j < write_begin + write_length; j++)
                {
                    auto image                      = image_write_deps.write_list[j];
                    auto next_gen                   = image_next_gen[image];
                    image_write_deps.generations[j] = next_gen;
                    image_next_gen[image]           = next_gen + 1;
                }

                // compute buffer generations that current pass reads.
                auto buf_read_begin  = buffer_read_deps.begins[current_pass];
                auto buf_read_length = buffer_read_deps.lengthes[current_pass];
                for (auto j = buf_read_begin; j < buf_read_begin + buf_read_length; j++)
                {
                    auto buffer                     = buffer_read_deps.read_list[j];
                    auto next_gen                   = buffer_next_gen[buffer];
                    buffer_read_deps.generations[j] = (next_gen == 0) ? 0 : (next_gen - 1);
                }

                // compute buffer generations that current pass writes.
                auto buf_write_begin  = buffer_write_deps.begins[current_pass];
                auto buf_write_length = buffer_write_deps.lengthes[current_pass];
                for (auto j = buf_write_begin; j < buf_write_begin + buf_write_length; j++)
                {
                    auto buffer                      = buffer_write_deps.write_list[j];
                    auto next_gen                    = buffer_next_gen[buffer];
                    buffer_write_deps.generations[j] = next_gen;
                    buffer_next_gen[buffer]          = next_gen + 1;
                }
            }

            // Step D: Culling
            // Analyze dependencies and mark passes as active/inactive

            active_pass_flags.assign(pass_count, false);
            std::queue<pass_handle> worklist;
            const auto invalid_pass = std::numeric_limits<pass_handle>::max();

            auto enqueue_pass = [&](pass_handle pass)
            {
                if (pass == invalid_pass || pass >= pass_count)
                {
                    return;
                }
                if (!active_pass_flags[pass])
                {
                    active_pass_flags[pass] = true;
                    worklist.push(pass);
                }
            };

            // Seed roots from declared outputs (images/buffers)
            for (const auto output_image : output_table.image_outputs)
            {
                if (output_image < producer_lookup_table.img_proc_map.size())
                {
                    enqueue_pass(producer_lookup_table.img_proc_map[output_image]);
                }
            }
            for (const auto output_buffer : output_table.buffer_outputs)
            {
                if (output_buffer < producer_lookup_table.buf_proc_map.size())
                {
                    enqueue_pass(producer_lookup_table.buf_proc_map[output_buffer]);
                }
            }

            // Reverse traversal: if a live pass reads a resource, its producer must be live.
            while (!worklist.empty())
            {
                const auto current_pass = worklist.front();
                worklist.pop();

                // image read dependencies
                {
                    const auto read_begin  = image_read_deps.begins[current_pass];
                    const auto read_length = image_read_deps.lengthes[current_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        const auto image = image_read_deps.read_list[j];
                        if (image < producer_lookup_table.img_proc_map.size())
                        {
                            enqueue_pass(producer_lookup_table.img_proc_map[image]);
                        }
                    }
                }

                // buffer read dependencies
                {
                    const auto read_begin  = buffer_read_deps.begins[current_pass];
                    const auto read_length = buffer_read_deps.lengthes[current_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        const auto buffer = buffer_read_deps.read_list[j];
                        if (buffer < producer_lookup_table.buf_proc_map.size())
                        {
                            enqueue_pass(producer_lookup_table.buf_proc_map[buffer]);
                        }
                    }
                }
            }

            // Step E: Validation (Recommended)
            // Validate graph correctness early and fail fast in debug builds.
            // Typical checks:
            // - Read-before-write on non-imported resources (producer == invalid_pass)
            // - Out-of-range resource handles in deps lists
            // - Multiple writers to same logical resource (until versioning is implemented)
            // - Empty output set (no roots) -> everything will be culled
            // - Cycles (once dependency edges exist)

            // Step F: DAG Construction (Not yet implemented)
            // Build pass-to-pass edges based on read dependencies and producer lookup:
            // - For each live pass P and each resource R in P.read_list:
            //   producer = proc_map[R]; if producer valid and producer != P => add edge producer -> P
            // Output:
            // - adjacency list (or CSR) for passes
            // - in-degree counts for topo sort

            // Step G: Scheduling / Topological Order (Not yet implemented)
            // Compute execution order for live passes (Kahn / DFS topo sort).
            // - If cycle detected => error
            // Store:
            // - a vector of sorted live passes
            // Future:
            // - parallel layers (no-dependency groups)

            // Step H: Resource Versioning (Not yet implemented)
            // Turn "multiple writes to same handle" into versions (handle, generation/version).
            // This makes:
            // - dependency analysis precise
            // - lifetime analysis correct
            // - aliasing possible
            // Note: you already compute generations per dep entry; next step is to promote that into
            // a stable "logical resource version" identity used throughout compile/execute.

            // Step I: Lifetime Analysis & Aliasing (Not yet implemented)
            // For each resource version, compute first/last use across the scheduled pass order.
            // Use this to:
            // - allocate transient resources from pools
            // - alias memory for non-overlapping lifetimes

            // Step J: Physical Resource Allocation (Not yet implemented)
            // Create actual GPU resources for live, non-imported resources.
            // - Filter out culled passes and unused resources
            // - Imported resources: do not create; expect bind_imported_* later (frame loop)
            // - Call backend to create/realize resources (possibly from pools)

            // Step K: Synchronization / Barrier Planning (Not yet implemented)
            // Build per-pass barrier lists based on resource access transitions between passes.
            // - Vulkan: layout transitions + src/dst stage/access
            // - DX12: resource state transitions
            // Output:
            // - per-pass barrier plan consumed by execute()

            // Step L: Execute Plan Build (Not yet implemented)
            // Finalize everything execute() needs:
            // - ordered pass list
            // - per-pass execute context (resolved resource handles)
            // - per-pass barrier plan
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
