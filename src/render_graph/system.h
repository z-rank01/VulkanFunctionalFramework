#pragma once

#include <algorithm>
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

        // Versioned dependency views generated during compile().
        // These are compile-time/internal and are derived from *_deps + versioning rules.
        std::vector<resource_version_handle> img_ver_read_handles;
        std::vector<resource_version_handle> img_ver_write_handles;
        std::vector<resource_version_handle> buf_ver_read_handles;
        std::vector<resource_version_handle> buf_ver_write_handles;

        version_producer_map producer_lookup_table;
        output_table output_table;

        // pass related
        graph_topology graph;
        directed_acyclic_graph dag;
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
            const auto invalid_pass = std::numeric_limits<pass_handle>::max();

            // Reset dependency storage
            image_read_deps.read_list.clear();
            image_read_deps.begins.assign(pass_count, 0);
            image_read_deps.lengthes.assign(pass_count, 0);
            image_read_deps.image_usages.clear();
            image_read_deps.buffer_usages.clear();
            image_write_deps.write_list.clear();
            image_write_deps.begins.assign(pass_count, 0);
            image_write_deps.lengthes.assign(pass_count, 0);
            image_write_deps.image_usages.clear();
            image_write_deps.buffer_usages.clear();
            buffer_read_deps.read_list.clear();
            buffer_read_deps.begins.assign(pass_count, 0);
            buffer_read_deps.lengthes.assign(pass_count, 0);
            buffer_read_deps.image_usages.clear();
            buffer_read_deps.buffer_usages.clear();
            buffer_write_deps.write_list.clear();
            buffer_write_deps.begins.assign(pass_count, 0);
            buffer_write_deps.lengthes.assign(pass_count, 0);
            buffer_write_deps.image_usages.clear();
            buffer_write_deps.buffer_usages.clear();
            output_table.image_outputs.clear();
            output_table.buffer_outputs.clear();

            img_ver_read_handles.clear();
            img_ver_write_handles.clear();
            buf_ver_read_handles.clear();
            buf_ver_write_handles.clear();

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

            const auto image_count  = meta_table.image_metas.names.size();
            const auto buffer_count = meta_table.buffer_metas.names.size();

            // Step B: Compute Resource Version (pack handle + version)
            // User-facing setup stage uses resource_handle only.
            // Here we derive a versioned view for internal compile-time algorithms.
            // IMPORTANT:
            // - Never use resource_version_handle (packed u64) as a vector index.
            // - Only index SoA arrays by resource_handle (low 32 bits).
            // - Read: graph.passes, *_deps
            // - Write: *_versions

            img_ver_read_handles.resize(image_read_deps.read_list.size());
            img_ver_write_handles.resize(image_write_deps.write_list.size());
            buf_ver_read_handles.resize(buffer_read_deps.read_list.size());
            buf_ver_write_handles.resize(buffer_write_deps.write_list.size());

            std::vector<version_handle> image_next_versions(image_count, 0);
            std::vector<version_handle> buffer_next_versions(buffer_count, 0);

            for (size_t i = 0; i < pass_count; i++)
            {
                const auto current_pass = graph.passes[i];

                // image reads
                {
                    const auto read_begin  = image_read_deps.begins[current_pass];
                    const auto read_length = image_read_deps.lengthes[current_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        const auto image = image_read_deps.read_list[j];
                        const auto next_version  = (image < image_next_versions.size()) ? image_next_versions[image] : 0;
                        if (next_version == 0)
                        {
                            // Unwritten (or imported-only) at this point; treat as having no producer.
                            // Validation should catch illegal read-before-write for non-imported resources.
                            img_ver_read_handles[j] = invalid_resource_version;
                        }
                        else
                        {
                            const auto version = static_cast<version_handle>(next_version - 1);
                            img_ver_read_handles[j] = pack(image, version);
                        }
                    }
                }

                // image writes
                {
                    const auto write_begin  = image_write_deps.begins[current_pass];
                    const auto write_length = image_write_deps.lengthes[current_pass];
                    for (auto j = write_begin; j < write_begin + write_length; j++)
                    {
                        const auto image = image_write_deps.write_list[j];
                        if (image >= image_count)
                        {
                            img_ver_write_handles[j] = invalid_resource_version;
                            continue;
                        }
                        const auto next_version = image_next_versions[image];
                        img_ver_write_handles[j] = pack(image, next_version);
                        image_next_versions[image] = static_cast<version_handle>(next_version + 1);
                    }
                }

                // buffer reads
                {
                    const auto read_begin  = buffer_read_deps.begins[current_pass];
                    const auto read_length = buffer_read_deps.lengthes[current_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        const auto buffer = buffer_read_deps.read_list[j];
                        const auto next_version   = (buffer < buffer_next_versions.size()) ? buffer_next_versions[buffer] : 0;
                        if (next_version == 0)
                        {
                            buf_ver_read_handles[j] = invalid_resource_version;
                        }
                        else
                        {
                            const auto version = static_cast<version_handle>(next_version - 1);
                            buf_ver_read_handles[j] = pack(buffer, version);
                        }
                    }
                }

                // buffer writes
                {
                    const auto write_begin  = buffer_write_deps.begins[current_pass];
                    const auto write_length = buffer_write_deps.lengthes[current_pass];
                    for (auto j = write_begin; j < write_begin + write_length; j++)
                    {
                        const auto buffer = buffer_write_deps.write_list[j];
                        if (buffer >= buffer_count)
                        {
                            buf_ver_write_handles[j] = invalid_resource_version;
                            continue;
                        }
                        const auto next_version = buffer_next_versions[buffer];
                        buf_ver_write_handles[j] = pack(buffer, next_version);
                        buffer_next_versions[buffer] = static_cast<version_handle>(next_version + 1);
                    }
                }
            }

            // Step C: Build resource-producer map (+ latest version per handle)
            // Build version -> producer lookup in a flat array (DOD/SoA friendly):
            // - offsets are indexed by resource_handle
            // - producers are indexed by (offset + version)
            // - Read: graph.passes, image_write_deps, buffer_write_deps, *_write_versions
            // - Write: producer_lookup_table

            // Build image offsets + latest
            producer_lookup_table.img_version_offsets.assign(static_cast<size_t>(image_count) + 1, 0);
            producer_lookup_table.latest_img.assign(image_count, invalid_resource_version);
            {
                uint32_t running = 0;
                for (resource_handle image = 0; image < image_count; image++)
                {
                    producer_lookup_table.img_version_offsets[image] = running;
                    const auto version = image_next_versions[image];
                    if (version > 0)
                    {
                        producer_lookup_table.latest_img[image] = pack(image, static_cast<version_handle>(version - 1));
                    }
                    running = (running + static_cast<uint32_t>(version));
                }
                producer_lookup_table.img_version_offsets[image_count] = running;
                producer_lookup_table.img_version_producers.assign(running, invalid_pass);
            }

            // Build buffer offsets + latest
            producer_lookup_table.buf_version_offsets.assign(static_cast<size_t>(buffer_count) + 1, 0);
            producer_lookup_table.latest_buf.assign(buffer_count, invalid_resource_version);
            {
                uint32_t running = 0;
                for (resource_handle buffer = 0; buffer < buffer_count; buffer++)
                {
                    producer_lookup_table.buf_version_offsets[buffer] = running;
                    const auto version = buffer_next_versions[buffer];
                    if (version > 0)
                    {
                        producer_lookup_table.latest_buf[buffer] = pack(buffer, static_cast<version_handle>(version - 1));
                    }
                    running = (running + static_cast<uint32_t>(version));
                }
                producer_lookup_table.buf_version_offsets[buffer_count] = running;
                producer_lookup_table.buf_version_producers.assign(running, invalid_pass);
            }

            // Fill image producers for each (image, version)
            for (size_t i = 0; i < pass_count; i++)
            {
                const auto current_pass = graph.passes[i];
                const auto begin        = image_write_deps.begins[current_pass];
                const auto length       = image_write_deps.lengthes[current_pass];
                for (auto j = begin; j < begin + length; j++)
                {
                    const auto image_version_handle   = img_ver_write_handles[j];
                    const auto image   = unpack_to_resource(image_version_handle);
                    const auto version = unpack_to_version(image_version_handle);
                    if (image >= image_count)
                    {
                        continue;
                    }
                    const auto base = producer_lookup_table.img_version_offsets[image];
                    const auto end  = producer_lookup_table.img_version_offsets[image + 1];
                    const auto idx  = static_cast<uint32_t>(base + version);
                    if (idx < end)
                    {
                        producer_lookup_table.img_version_producers[idx] = current_pass;
                    }
                }
            }

            // Fill buffer producers for each (buffer, version)
            for (size_t i = 0; i < pass_count; i++)
            {
                const auto current_pass = graph.passes[i];
                const auto begin        = buffer_write_deps.begins[current_pass];
                const auto length       = buffer_write_deps.lengthes[current_pass];
                for (auto j = begin; j < begin + length; j++)
                {
                    const auto buffer_version_handle   = buf_ver_write_handles[j];
                    const auto buffer   = unpack_to_resource(buffer_version_handle);
                    const auto version = unpack_to_version(buffer_version_handle);
                    if (buffer >= buffer_count)
                    {
                        continue;
                    }
                    const auto base = producer_lookup_table.buf_version_offsets[buffer];
                    const auto end  = producer_lookup_table.buf_version_offsets[buffer + 1];
                    const auto idx  = static_cast<uint32_t>(base + version);
                    if (idx < end)
                    {
                        producer_lookup_table.buf_version_producers[idx] = current_pass;
                    }
                }
            }

            // Step D: Culling
            // Analyze dependencies and mark passes as active/inactive

            active_pass_flags.assign(pass_count, false);
            std::queue<pass_handle> culling_worklist;
            // invalid_pass is defined above for producer map.

            auto enqueue_pass = [&](pass_handle pass)
            {
                if (pass == invalid_pass || pass >= pass_count)
                {
                    return;
                }
                if (!active_pass_flags[pass])
                {
                    active_pass_flags[pass] = true;
                    culling_worklist.push(pass);
                }
            };

            auto enqueue_image_producer = [&](resource_version_handle version)
            {
                if (version == invalid_resource_version)
                {
                    return;
                }
                const auto image   = unpack_to_resource(version);
                const auto ver = unpack_to_version(version);
                if (image >= image_count)
                {
                    return;
                }
                const auto base = producer_lookup_table.img_version_offsets[image];
                const auto end  = producer_lookup_table.img_version_offsets[image + 1];
                const auto idx  = static_cast<uint32_t>(base + ver);
                if (idx >= end)
                {
                    return;
                }
                enqueue_pass(producer_lookup_table.img_version_producers[idx]);
            };

            auto enqueue_buffer_producer = [&](resource_version_handle version)
            {
                if (version == invalid_resource_version)
                {
                    return;
                }
                const auto buffer   = unpack_to_resource(version);
                const auto ver = unpack_to_version(version);
                if (buffer >= buffer_count)
                {
                    return;
                }
                const auto base = producer_lookup_table.buf_version_offsets[buffer];
                const auto end  = producer_lookup_table.buf_version_offsets[buffer + 1];
                const auto idx  = static_cast<uint32_t>(base + ver);
                if (idx >= end)
                {
                    return;
                }
                enqueue_pass(producer_lookup_table.buf_version_producers[idx]);
            };

            auto get_image_producer = [&](resource_version_handle version) -> pass_handle
            {
                if (version == invalid_resource_version)
                {
                    return invalid_pass;
                }
                const auto image_handle = unpack_to_resource(version);
                const auto ver          = unpack_to_version(version);
                if (image_handle >= image_count)
                {
                    return invalid_pass;
                }
                const auto base = producer_lookup_table.img_version_offsets[image_handle];
                const auto end  = producer_lookup_table.img_version_offsets[image_handle + 1];
                const auto idx  = static_cast<uint32_t>(base + ver);
                if (idx >= end)
                {
                    return invalid_pass;
                }
                return producer_lookup_table.img_version_producers[idx];
            };

            auto get_buffer_producer = [&](resource_version_handle version) -> pass_handle
            {
                if (version == invalid_resource_version)
                {
                    return invalid_pass;
                }
                const auto buffer_handle = unpack_to_resource(version);
                const auto ver           = unpack_to_version(version);
                if (buffer_handle >= buffer_count)
                {
                    return invalid_pass;
                }
                const auto base = producer_lookup_table.buf_version_offsets[buffer_handle];
                const auto end  = producer_lookup_table.buf_version_offsets[buffer_handle + 1];
                const auto idx  = static_cast<uint32_t>(base + ver);
                if (idx >= end)
                {
                    return invalid_pass;
                }
                return producer_lookup_table.buf_version_producers[idx];
            };

            // Seed roots from declared outputs (images/buffers)
            for (const auto output_image : output_table.image_outputs)
            {
                if (output_image < image_count)
                {
                    enqueue_image_producer(producer_lookup_table.latest_img[output_image]);
                }
            }
            for (const auto output_buffer : output_table.buffer_outputs)
            {
                if (output_buffer < buffer_count)
                {
                    enqueue_buffer_producer(producer_lookup_table.latest_buf[output_buffer]);
                }
            }

            // Reverse traversal: if a live pass reads a resource, its producer must be live.
            while (!culling_worklist.empty())
            {
                const auto current_pass = culling_worklist.front();
                culling_worklist.pop();

                // image read dependencies
                {
                    const auto read_begin  = image_read_deps.begins[current_pass];
                    const auto read_length = image_read_deps.lengthes[current_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        enqueue_image_producer(img_ver_read_handles[j]);
                    }
                }

                // buffer read dependencies
                {
                    const auto read_begin  = buffer_read_deps.begins[current_pass];
                    const auto read_length = buffer_read_deps.lengthes[current_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        enqueue_buffer_producer(buf_ver_read_handles[j]);
                    }
                }
            }

            // Step E: Validate Resource
            // Validate graph correctness early and fail fast in debug builds.
            // Typical checks:
            // - Read-before-write on non-imported resources (producer == invalid_pass)
            // - Out-of-range resource handles in deps lists
            // - Empty output set (no roots) -> everything will be culled
            
            // check outputs
            assert((!output_table.image_outputs.empty() || !output_table.buffer_outputs.empty()) && "Error: No outputs declared");

            // check read-before-write issues and out-of-range handles
            for (size_t i = 0; i < pass_count; i++)
            {
                if (!active_pass_flags[i]) continue;

                const auto current_pass = graph.passes[i];
                // image reads
                {
                    const auto read_begin  = image_read_deps.begins[current_pass];
                    const auto read_length = image_read_deps.lengthes[current_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        const auto image_handle = image_read_deps.read_list[j];
                        if (image_handle >= image_count)
                        {
                            assert(false && "Error: Image read out-of-range detected!");
                        }

                        const auto version_handle = img_ver_read_handles[j];
                        const bool is_imported    = meta_table.image_metas.is_imported[image_handle];
                        const auto producer       = get_image_producer(version_handle);

                        if (version_handle == invalid_resource_version)
                        {
                            // next_version==0: no internal write happened before this read.
                            // This is only legal for imported resources.
                            assert(is_imported && "Error: Image read-before-write detected!");
                        }
                        else
                        {
                            assert((is_imported || producer != invalid_pass) && "Error: Image read-before-write detected!");
                        }
                    }
                }
                // buffer reads
                {
                    const auto read_begin  = buffer_read_deps.begins[current_pass];
                    const auto read_length = buffer_read_deps.lengthes[current_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        const auto buffer_handle = buffer_read_deps.read_list[j];
                        if (buffer_handle >= buffer_count)
                        {
                            assert(false && "Error: Buffer read out-of-range detected!");
                        }

                        const auto version_handle = buf_ver_read_handles[j];
                        const bool is_imported    = meta_table.buffer_metas.is_imported[buffer_handle];
                        const auto producer       = get_buffer_producer(version_handle);

                        if (version_handle == invalid_resource_version)
                        {
                            assert(is_imported && "Error: Buffer read-before-write detected!");
                        }
                        else
                        {
                            assert((is_imported || producer != invalid_pass) && "Error: Buffer read-before-write detected!");
                        }
                    }
                }
                // image writes
                {
                    const auto write_begin  = image_write_deps.begins[current_pass];
                    const auto write_length = image_write_deps.lengthes[current_pass];
                    for (auto j = write_begin; j < write_begin + write_length; j++)
                    {
                        const auto image_handle = image_write_deps.write_list[j];
                        assert(image_handle < image_count && "Error: Image write out-of-range detected!");
                        assert(img_ver_write_handles[j] != invalid_resource_version && "Error: Image write out-of-range detected!");
                    }
                }
                // buffer writes
                {
                    const auto write_begin  = buffer_write_deps.begins[current_pass];
                    const auto write_length = buffer_write_deps.lengthes[current_pass];
                    for (auto j = write_begin; j < write_begin + write_length; j++)
                    {
                        const auto buffer_handle = buffer_write_deps.write_list[j];
                        assert(buffer_handle < buffer_count && "Error: Buffer write out-of-range detected!");
                        assert(buf_ver_write_handles[j] != invalid_resource_version && "Error: Buffer write out-of-range detected!");
                    }
                }
            }

            // Step F: DAG Construction (Not yet implemented)
            // Build pass-to-pass edges based on read dependencies and producer lookup:
            // - For each live pass P and each resource R in P.read_list:
            //   producer = proc_map[R]; if producer valid and producer != P => add edge producer -> P
            // Output:
            // - adjacency list (or CSR) for passes
            // - in-degree counts for topo sort

            // Forward adjacency (CSR): producer -> consumer.
            // We build edges for all active passes (already culled from declared outputs).
            std::vector<std::vector<pass_handle>> outgoing(pass_count);
            auto add_edge = [&](pass_handle from, pass_handle to)
            {
                if (from == invalid_pass || to == invalid_pass)
                {
                    return;
                }
                if (from >= pass_count || to >= pass_count)
                {
                    return;
                }
                if (from == to)
                {
                    return;
                }
                if (!active_pass_flags[from] || !active_pass_flags[to])
                {
                    return;
                }
                outgoing[from].push_back(to);
            };

            for (size_t i = 0; i < pass_count; i++)
            {
                const auto consumer_pass = graph.passes[i];
                if (!active_pass_flags[consumer_pass])
                {
                    continue;
                }

                // image read dependencies: producer(img_ver_read) -> consumer
                {
                    const auto read_begin  = image_read_deps.begins[consumer_pass];
                    const auto read_length = image_read_deps.lengthes[consumer_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        const auto producer = get_image_producer(img_ver_read_handles[j]);
                        add_edge(producer, consumer_pass);
                    }
                }

                // buffer read dependencies: producer(buf_ver_read) -> consumer
                {
                    const auto read_begin  = buffer_read_deps.begins[consumer_pass];
                    const auto read_length = buffer_read_deps.lengthes[consumer_pass];
                    for (auto j = read_begin; j < read_begin + read_length; j++)
                    {
                        const auto producer = get_buffer_producer(buf_ver_read_handles[j]);
                        add_edge(producer, consumer_pass);
                    }
                }
            }

            dag.adjacency_list.clear();
            dag.adjacency_begins.assign(static_cast<size_t>(pass_count) + 1, 0);
            dag.in_degrees.assign(pass_count, 0);
            dag.out_degrees.assign(pass_count, 0);

            // De-duplicate edges per producer and compute degrees.
            for (pass_handle pass = 0; pass < pass_count; pass++)
            {
                auto& list = outgoing[pass];
                std::sort(list.begin(), list.end());
                list.erase(std::unique(list.begin(), list.end()), list.end());
            }
            for (pass_handle from = 0; from < pass_count; from++)
            {
                dag.out_degrees[from] = static_cast<uint32_t>(outgoing[from].size());
                for (const auto dst_pass : outgoing[from])
                {
                    dag.in_degrees[dst_pass]++;
                }
            }

            // Build CSR arrays.
            uint32_t running = 0;
            for (pass_handle from = 0; from < pass_count; from++)
            {
                dag.adjacency_begins[from] = running;
                const auto& list = outgoing[from];
                dag.adjacency_list.insert(dag.adjacency_list.end(), list.begin(), list.end());
                running = static_cast<uint32_t>(dag.adjacency_list.size());
            }
            dag.adjacency_begins[pass_count] = running;


            // Step G: Validate DAG (Not yet implemented)
            // Validate constructed DAG:
            // - No cycles (would block scheduling)
            // - All live passes reachable from roots (should be guaranteed by culling)

            // Step G: Scheduling / Topological Order (Not yet implemented)
            // Compute execution order for live passes (Kahn / DFS topo sort).
            // - If cycle detected => error
            // Store:
            // - a vector of sorted live passes
            // Future:
            // - parallel layers (no-dependency groups)

            // Step H: Lifetime Analysis & Aliasing (Not yet implemented)
            // For each resource version, compute first/last use across the scheduled pass order.
            // Use this to:
            // - allocate transient resources from pools
            // - alias memory for non-overlapping lifetimes

            // Step I: Physical Resource Allocation (Not yet implemented)
            // Create actual GPU resources for live, non-imported resources.
            // - Filter out culled passes and unused resources
            // - Imported resources: do not create; expect bind_imported_* later (frame loop)
            // - Call backend to create/realize resources (possibly from pools)

            // Step J: Synchronization / Barrier Planning (Not yet implemented)
            // Build per-pass barrier lists based on resource access transitions between passes.
            // - Vulkan: layout transitions + src/dst stage/access
            // - DX12: resource state transitions
            // Output:
            // - per-pass barrier plan consumed by execute()

            // Step K: Execute Plan Build (Not yet implemented)
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
