#include "simple_camera.h"

#include <glm/ext/matrix_transform.hpp>


#define GLM_ENABLE_EXPERIMENTAL
#include <algorithm>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtx/norm.hpp>


namespace interface
{
    using glm::mat4;

    simple_camera::simple_camera()
    {
        // initialize camera attributes

        camera_data.position          = glm::vec3(0.0F, 0.0F, 10.0F);
        camera_data.yaw               = -90.0F;
        camera_data.pitch             = 0.0F;
        camera_data.wheel_speed       = 0.1F;
        camera_data.movement_speed    = 10.0F;
        camera_data.mouse_sensitivity = 0.2F;
        camera_data.zoom              = 45.0F;
        camera_data.world_up          = glm::vec3(0.0F, 1.0F, 0.0F); // Y-axis is up in Vulkan
        camera_data.aspect_ratio      = 16.0F / 9.0F;
        camera_data.width             = 1600;

        // initialize camera basic vectors

        camera_data.front = glm::vec3(0.0F, 0.0F, -1.0F); // look at -z direction
        camera_data.right = glm::vec3(1.0F, 0.0F, 0.0F);  // the right direction is +x
        camera_data.up    = glm::vec3(0.0F, 1.0F, 0.0F);  // up direction is +y (because Y-axis is up in Vulkan)

        // initialize focus point related parameters

        camera_data.focus_point        = glm::vec3(0.0F); // default focus on origin
        camera_data.has_focus_point    = true;            // default enables focus point
        camera_data.focus_distance     = 10.0F;           // default focus distance
        camera_data.min_focus_distance = 0.5F;            // minimum focus distance
        camera_data.max_focus_distance = 10000.0F;        // maximum focus distance

        // initialize time point

        last_tick_time = std::chrono::high_resolution_clock::now();
    }

    void simple_camera::tick(const InputEvent& event)
    {
        process_mouse_input(event);
        process_keyboard_input(event);
        camera_data.update_camera_vectors();
    }

    glm::mat4 simple_camera::get_matrix(transform_matrix_type matrix_type) const
    {
        switch (matrix_type)
        {
        case transform_matrix_type::model:
            return {1.0F};
        case transform_matrix_type::view:
            return glm::lookAt(camera_data.position, camera_data.position + camera_data.front, camera_data.up);
        case transform_matrix_type::projection:
            return glm::perspective(glm::radians(camera_data.zoom), camera_data.aspect_ratio, 0.1F, 1000.0F);
        default:
            return {1.0F};
        }
    }

    void simple_camera::process_keyboard_input(const interface::InputEvent& /*event*/)
    {
        // 1. Calculate Delta Time
        auto current_tick_time = std::chrono::high_resolution_clock::now();
        float delta_time       = std::chrono::duration<float>(current_tick_time - last_tick_time).count();
        last_tick_time         = current_tick_time;

        // Prevent huge delta time jump
        if (delta_time > 0.1F) delta_time = 0.016F;

        // 2. Helper Lambda: Map keys to axis value (-1.0, 0.0, 1.0)
        auto get_axis = [&](KeyCode pos_key, KeyCode neg_key, KeyCode alt_pos = KeyCode::Unknown, KeyCode alt_neg = KeyCode::Unknown)
        {
            float val = 0.0F;
            if (pressed_keys.contains(pos_key) || (alt_pos != KeyCode::Unknown && pressed_keys.contains(alt_pos)))
                val += 1.0F;
            if (pressed_keys.contains(neg_key) || (alt_neg != KeyCode::Unknown && pressed_keys.contains(alt_neg)))
                val -= 1.0F;
            return val;
        };

        // 3. Calculate Input Direction Vector (Local Space)
        // x: Right(+)/Left(-), y: Up(+)/Down(-), z: Front(+)/Back(-)
        glm::vec3 input_dir(0.0F);

        if (free_look_mode)
        {
            // Free Look Mapping: W/S->Front, A/D->Right, E/Q->Up(Vertical)
            input_dir.x = get_axis(KeyCode::D, KeyCode::A, KeyCode::Right, KeyCode::Left);
            input_dir.y = get_axis(KeyCode::E, KeyCode::Q);
            input_dir.z = get_axis(KeyCode::W, KeyCode::S, KeyCode::Up, KeyCode::Down);
        }
        else
        {
            // Normal Mapping: D/A->X, W/S->Y, E/Q->Z (Note: Q is +Z which is Back, E is -Z which is Front)
            input_dir.x = get_axis(KeyCode::D, KeyCode::A, KeyCode::Right, KeyCode::Left);
            input_dir.y = get_axis(KeyCode::W, KeyCode::S, KeyCode::Up, KeyCode::Down);
            input_dir.z = get_axis(KeyCode::Q, KeyCode::E); // Q moves back (+Z), E moves front (-Z)
        }

        // 4. Update Speed Acceleration
        bool is_moving = glm::length2(input_dir) > 0.01F;
        if (is_moving)
        {
            current_speed_multiplier = std::min(current_speed_multiplier + (acceleration_rate * delta_time), max_speed_multiplier);
        }
        else
        {
            current_speed_multiplier = 1.0F;
            return; // No movement, early exit
        }

        // 5. Apply Movement
        float velocity = camera_data.movement_speed * current_speed_multiplier;
        float step = velocity * delta_time; // Apply delta_time here for frame-rate independence, physhically correct

        if (free_look_mode)
        {
            // Calculate distance scale for focus constraint
            float distance_scale = 1.0F;
            if (camera_data.has_focus_point && camera_data.focus_constraint_enabled)
            {
                float dist     = glm::length(camera_data.position - camera_data.focus_point);
                distance_scale = std::clamp(dist / camera_data.focus_distance,
                                            camera_data.min_focus_distance / camera_data.focus_distance,
                                            camera_data.max_focus_distance / camera_data.focus_distance);
            }

            float current_step = step / distance_scale;

            // Apply local directions to world position
            camera_data.position += camera_data.right * input_dir.x * current_step;
            camera_data.position += camera_data.up * input_dir.y * current_step; // World Up or Camera Up? Original used camera_data.up
            camera_data.position += camera_data.front * input_dir.z * current_step;
        }
        else
        {
            // Direct world axis movement
            camera_data.position += input_dir * step;
        }
    }

    void simple_camera::process_mouse_input(const interface::InputEvent& event)
    {
        // Helper for Free Look Logic

        auto handle_free_look = [&](float x_offset, float y_offset)
        {
            float sensitivity_scale = 1.0F;
            if (camera_data.has_focus_point && camera_data.focus_constraint_enabled)
            {
                float dist        = glm::length(camera_data.position - camera_data.focus_point);
                sensitivity_scale = std::clamp(dist / camera_data.focus_distance,
                                               camera_data.min_focus_distance / camera_data.focus_distance,
                                               camera_data.max_focus_distance / camera_data.focus_distance);
            }

            camera_data.yaw += x_offset * camera_data.mouse_sensitivity * sensitivity_scale;
            camera_data.pitch += y_offset * camera_data.mouse_sensitivity * sensitivity_scale;
            camera_data.pitch = std::clamp(camera_data.pitch, -89.0F, 89.0F);
        };

        // Helper for Pan Logic

        auto handle_pan = [&](float x_offset, float y_offset)
        {
            float dist = camera_data.has_focus_point ? glm::length(camera_data.position - camera_data.focus_point) : camera_data.focus_distance;
            float distance_scale = std::clamp(dist / camera_data.focus_distance,
                                              camera_data.min_focus_distance / camera_data.focus_distance,
                                              camera_data.max_focus_distance / camera_data.focus_distance);

            constexpr float base_pan_speed = 0.005F;
            float pan_speed                = camera_data.focus_constraint_enabled ? base_pan_speed / distance_scale : base_pan_speed;
            float move_speed               = camera_data.movement_speed * pan_speed;

            camera_data.position -= camera_data.right * x_offset * move_speed;
            camera_data.position += camera_data.up * y_offset * move_speed;
        };

        // Process Event

        switch (event.type)
        {
        case EventType::KeyDown:
            pressed_keys.insert(event.key.key);
            break;
        case EventType::KeyUp:
            pressed_keys.erase(event.key.key);
            break;

        case EventType::MouseButtonDown:
            last_x = event.mouse_button.x;
            last_y = event.mouse_button.y;
            if (event.mouse_button.button == MouseButton::Right)
                free_look_mode = true;
            if (event.mouse_button.button == MouseButton::Middle)
                camera_pan_mode = true;
            break;

        case EventType::MouseButtonUp:
            if (event.mouse_button.button == MouseButton::Right)
                free_look_mode = false;
            if (event.mouse_button.button == MouseButton::Middle)
                camera_pan_mode = false;
            break;

        case EventType::MouseMove:
            if (free_look_mode || camera_pan_mode)
            {
                float x_offset = event.mouse_move.x - last_x;
                float y_offset = last_y - event.mouse_move.y; // Reversed since y-coordinates go from bottom to top
                last_x         = event.mouse_move.x;
                last_y         = event.mouse_move.y;

                if (free_look_mode)
                    handle_free_look(x_offset, y_offset);
                if (camera_pan_mode)
                    handle_pan(x_offset, y_offset);
            }
            break;

        case EventType::MouseWheel:
        {
            float offset = event.mouse_wheel.y;
            if (camera_data.has_focus_point && camera_data.focus_constraint_enabled)
            {
                float dist  = glm::length(camera_data.position - camera_data.focus_point);
                float scale = std::clamp(dist / camera_data.focus_distance,
                                         camera_data.min_focus_distance / camera_data.focus_distance,
                                         camera_data.max_focus_distance / camera_data.focus_distance);

                float zoom_step = camera_data.movement_speed * 0.5F / scale;
                camera_data.position += camera_data.front * (offset > 0 ? zoom_step : -zoom_step);
            }
            else
            {
                camera_data.zoom = std::clamp(camera_data.zoom - offset, 1.0F, 45.0F);
            }
            break;
        }
        default:
            break;
        }
    }

    std::any simple_camera::get_impl_internal(camera_attribute attribute) const
    {
        switch (attribute)
        {
        case camera_attribute::movement_speed:
            return camera_data.movement_speed;
        case camera_attribute::zoom:
            return camera_data.zoom;
        case camera_attribute::width:
            return camera_data.width;
        case camera_attribute::aspect_ratio:
            return camera_data.aspect_ratio;
        case camera_attribute::position:
            return camera_data.position;
        case camera_attribute::mouse_sensitivity:
            return camera_data.mouse_sensitivity;
        case camera_attribute::wheel_speed:
            return camera_data.wheel_speed;
        case camera_attribute::focus_point:
            return camera_data.focus_point;
        case camera_attribute::focus_distance:
            return camera_data.focus_distance;
        case camera_attribute::has_focus_point:
            return camera_data.has_focus_point;
        case camera_attribute::focus_constraint_enabled:
            return camera_data.focus_constraint_enabled;
        default:
            return {}; // Return empty std::any for unknown attributes
        }
    }

    void simple_camera::set_impl_internal(camera_attribute attribute, const std::any& value)
    {
        switch (attribute)
        {
        case camera_attribute::movement_speed:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.movement_speed = *val;
            break;
        case camera_attribute::width:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.width = *val;
            break;
        case camera_attribute::aspect_ratio:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.aspect_ratio = *val;
            break;
        case camera_attribute::mouse_sensitivity:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.mouse_sensitivity = *val;
            break;
        case camera_attribute::wheel_speed:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.wheel_speed = *val;
            break;
        case camera_attribute::focus_point:
            if (const auto* val = std::any_cast<glm::vec3>(&value))
                camera_data.focus_point = *val;
            break;
        case camera_attribute::focus_distance:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.focus_distance = *val;
            break;
        case camera_attribute::has_focus_point:
            if (const auto* val = std::any_cast<bool>(&value))
                camera_data.has_focus_point = *val;
            break;
        case camera_attribute::focus_constraint_enabled:
            if (const auto* val = std::any_cast<bool>(&value))
                camera_data.focus_constraint_enabled = *val;
            break;
        case camera_attribute::zoom:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.zoom = *val;
            break;
        case camera_attribute::position:
            if (const auto* val = std::any_cast<glm::vec3>(&value))
                camera_data.position = *val;
            break;
        default:
            break;
        }
    }
}; // namespace interface
