#pragma once

namespace render_graph::unit_test
{
    // Builds a small graph with:
    // - multiple images & buffers
    // - resources written by later passes (overwrite)
    // - an imported resource that is only read (no producer)
    // Then invokes render_graph_system::compile() and lets you inspect:
    // - render_graph_system::img_proc_map / buf_proc_map
    // - expected mappings stored in debugger-visible state
    void resource_producer_map_compile_test();
}
