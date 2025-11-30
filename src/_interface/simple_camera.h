#pragma once
#include <chrono>
#include <glm/glm.hpp>
#include <unordered_set>

#include "camera.h"
#include "input.h"

namespace interface
{

    class simple_camera : public camera
    {
    public:
        simple_camera();
        ~simple_camera() override = default;

        void tick(const interface::input_event& event) override;
        glm::mat4 get_matrix(transform_matrix_type matrix_type) const override;

    private:
        camera_data camera_data;

        // Time based acceleration

        std::chrono::high_resolution_clock::time_point last_tick_time;
        float current_speed_multiplier = 1.0F;
        float max_speed_multiplier     = 20.0F; // Max 20x speed
        float acceleration_rate        = 10.0F; // Increase 10x per second

        // Input handling members

        float last_x         = 0.0F;
        float last_y         = 0.0F;
        float orbit_distance = 0.0F; // distance from center during orbit rotation
        std::unordered_set<interface::key_code> pressed_keys;
        bool free_look_mode  = false; // free look mode (right mouse button held)
        bool camera_pan_mode = false; // camera pan mode (middle mouse button)

        // Override private virtual functions from base class

        std::any get_impl_internal(camera_attribute attribute) const override;
        void set_impl_internal(camera_attribute attribute, const std::any& value) override;
        void process_keyboard_input(const interface::input_event& event);
        void process_mouse_input(const interface::input_event& event);
    };
} // namespace interface
