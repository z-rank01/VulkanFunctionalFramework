#include "render_graph/unit_test/resource_producer_map_compile_test.h"

#include <limits>
#include <vector>

#include "render_graph/system.h"

namespace render_graph::unit_test
{
    namespace
    {
        struct test_state_t
        {
            // Images
            resource_handle img_a1            = 0;
            resource_handle img_a2            = 0;
            resource_handle img_b2            = 0;
            resource_handle img_swapchain     = 0;
            resource_handle img_external_only = 0; // created/imported, only read

            // Buffers
            resource_handle buf_b1 = 0;
            resource_handle buf_b3 = 0;

            // Expected producer maps (indexed by handle)
            std::vector<pass_handle> expected_img_proc;
            std::vector<pass_handle> expected_buf_proc;

            static pass_handle invalid_pass()  { return std::numeric_limits<pass_handle>::max(); }

            void reset()
            {
                *this = test_state_t{};
                expected_img_proc.clear();
                expected_buf_proc.clear();
            }

            void expect_img(resource_handle image, pass_handle producer)
            {
                if (expected_img_proc.size() <= image)
                {
                    expected_img_proc.resize(static_cast<size_t>(image) + 1, invalid_pass());
                }
                expected_img_proc[image] = producer;
            }

            void expect_buf(resource_handle buffer, pass_handle producer)
            {
                if (expected_buf_proc.size() <= buffer)
                {
                    expected_buf_proc.resize(static_cast<size_t>(buffer) + 1, invalid_pass());
                }
                expected_buf_proc[buffer] = producer;
            }
        };

        test_state_t& test_state()
        {
            static test_state_t state{};
            return state;
        }

        void noop_execute(pass_execute_context&) { }

        // Pass 0: create/write a1, a2, b1
        void pass_a_setup(pass_setup_context& ctx)
        {
            auto& state = test_state();

            state.img_a1 = ctx.create_image(image_info{
                .name          = "img_a1",
                .fmt           = format::R8G8B8A8_UNORM,
                .extent        = {.width = 256, .height = 256, .depth = 1},
                .usage         = image_usage::COLOR_ATTACHMENT,
                .type          = image_type::TYPE_2D,
                .flags         = image_flags::NONE,
                .mip_levels    = 1,
                .array_layers  = 1,
                .sample_counts = 1,
                .imported      = false,
            });
            ctx.write_image(state.img_a1, image_usage::COLOR_ATTACHMENT);
            state.expect_img(state.img_a1, ctx.current_pass);

            state.img_a2 = ctx.create_image(image_info{
                .name          = "img_a2",
                .fmt           = format::R8G8B8A8_UNORM,
                .extent        = {.width = 256, .height = 256, .depth = 1},
                .usage         = image_usage::COLOR_ATTACHMENT,
                .type          = image_type::TYPE_2D,
                .flags         = image_flags::NONE,
                .mip_levels    = 1,
                .array_layers  = 1,
                .sample_counts = 1,
                .imported      = false,
            });
            ctx.write_image(state.img_a2, image_usage::COLOR_ATTACHMENT);
            state.expect_img(state.img_a2, ctx.current_pass);

            state.buf_b1 = ctx.create_buffer(buffer_info{
                .name     = "buf_b1",
                .size     = 1024,
                .usage    = buffer_usage::NONE,
                .imported = false,
            });
            ctx.write_buffer(state.buf_b1, buffer_usage::STORAGE_BUFFER);
            state.expect_buf(state.buf_b1, ctx.current_pass);
        }

        // Pass 1: read a1, write b2, rewrite b1 (overwrite producer)
        void pass_b_setup(pass_setup_context& ctx)
        {
            auto& state = test_state();

            ctx.read_image(state.img_a1, image_usage::SAMPLED);

            state.img_b2 = ctx.create_image(image_info{
                .name          = "img_b2",
                .fmt           = format::R8G8B8A8_UNORM,
                .extent        = {.width = 256, .height = 256, .depth = 1},
                .usage         = image_usage::COLOR_ATTACHMENT,
                .type          = image_type::TYPE_2D,
                .flags         = image_flags::NONE,
                .mip_levels    = 1,
                .array_layers  = 1,
                .sample_counts = 1,
                .imported      = false,
            });
            ctx.write_image(state.img_b2, image_usage::COLOR_ATTACHMENT);
            state.expect_img(state.img_b2, ctx.current_pass);

            // Overwrite producer for buf_b1
            ctx.read_buffer(state.buf_b1, buffer_usage::STORAGE_BUFFER);
            ctx.write_buffer(state.buf_b1, buffer_usage::STORAGE_BUFFER);
            state.expect_buf(state.buf_b1, ctx.current_pass);
        }

        // Pass 2: read b2 & b1, rewrite a2 (overwrite producer), create/write b3
        void pass_c_setup(pass_setup_context& ctx)
        {
            auto& state = test_state();

            ctx.read_image(state.img_b2, image_usage::SAMPLED);
            ctx.read_buffer(state.buf_b1, buffer_usage::STORAGE_BUFFER);

            // Rewrite producer for img_a2
            ctx.write_image(state.img_a2, image_usage::COLOR_ATTACHMENT);
            state.expect_img(state.img_a2, ctx.current_pass);

            state.buf_b3 = ctx.create_buffer(buffer_info{
                .name     = "buf_b3",
                .size     = 2048,
                .usage    = buffer_usage::NONE,
                .imported = false,
            });
            ctx.write_buffer(state.buf_b3, buffer_usage::STORAGE_BUFFER);
            state.expect_buf(state.buf_b3, ctx.current_pass);
        }

        // Pass 3: imported external image that is only read (no write) -> producer should remain invalid
        void pass_external_input_setup(pass_setup_context& ctx)
        {
            auto& state = test_state();

            state.img_external_only = ctx.create_image(image_info{
                .name          = "img_external_only",
                .fmt           = format::R8G8B8A8_UNORM,
                .extent        = {.width = 64, .height = 64, .depth = 1},
                .usage         = image_usage::SAMPLED,
                .type          = image_type::TYPE_2D,
                .flags         = image_flags::NONE,
                .mip_levels    = 1,
                .array_layers  = 1,
                .sample_counts = 1,
                .imported      = true,
            });

            ctx.read_image(state.img_external_only, image_usage::SAMPLED);
            state.expect_img(state.img_external_only, state.invalid_pass());
        }

        // Pass 4: read a2 & external, write imported swapchain
        void pass_present_setup(pass_setup_context& ctx)
        {
            auto& state = test_state();

            ctx.read_image(state.img_a2, image_usage::SAMPLED);
            ctx.read_image(state.img_external_only, image_usage::SAMPLED);

            state.img_swapchain = ctx.create_image(image_info{
                .name          = "swapchain_backbuffer_test",
                .fmt           = format::R8G8B8A8_UNORM,
                .extent        = {.width = 256, .height = 256, .depth = 1},
                .usage         = image_usage::COLOR_ATTACHMENT,
                .type          = image_type::TYPE_2D,
                .flags         = image_flags::NONE,
                .mip_levels    = 1,
                .array_layers  = 1,
                .sample_counts = 1,
                .imported      = true,
            });

            ctx.write_image(state.img_swapchain, image_usage::COLOR_ATTACHMENT);
            state.expect_img(state.img_swapchain, ctx.current_pass);
        }
    } // namespace

    void resource_producer_map_compile_test()
    {
        auto& state = test_state();
        state.reset();

        render_graph_system system;

        system.add_pass(pass_a_setup, noop_execute);
        system.add_pass(pass_b_setup, noop_execute);
        system.add_pass(pass_c_setup, noop_execute);
        system.add_pass(pass_external_input_setup, noop_execute);
        system.add_pass(pass_present_setup, noop_execute);

        system.compile();

        // Set a breakpoint here and inspect:
        // - system.img_proc_map / system.buf_proc_map
        // - test_state().expected_img_proc / expected_buf_proc
        // Also pay attention to rewritten resources (img_a2, buf_b1).
        (void)system;
    }
} // namespace render_graph::unit_test
