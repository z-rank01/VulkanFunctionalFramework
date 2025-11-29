#include "simple_camera.h"

#include <algorithm>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

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
        camera_data.movement_speed    = 0.05F;
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

        last_tick_time = std::chrono::high_resolution_clock::now();
    }

    void simple_camera::tick(const InputEvent& event)
    {
        switch (event.type)
        {
        case EventType::Quit:
            break;

            // keyboard event

        case EventType::KeyDown:
            pressed_keys.insert(event.key.key);
            break;
        case EventType::KeyUp:
            pressed_keys.erase(event.key.key);
            break;

            // mouse event

        case EventType::MouseButtonDown:
            last_x          = event.mouse_button.x;
            last_y          = event.mouse_button.y;
            free_look_mode  = event.mouse_button.button == MouseButton::Right ? true : free_look_mode;
            camera_pan_mode = event.mouse_button.button == MouseButton::Middle ? true : camera_pan_mode;
            break;
        case EventType::MouseButtonUp:
            free_look_mode  = event.mouse_button.button == MouseButton::Right ? false : free_look_mode;
            camera_pan_mode = event.mouse_button.button == MouseButton::Middle ? false : camera_pan_mode;
            break;
        case EventType::MouseMove:
            if (!(free_look_mode || camera_pan_mode))
                break;
            {
                const float x_pos    = event.mouse_move.x;
                const float y_pos    = event.mouse_move.y;
                const float x_offset = x_pos - last_x;
                const float y_offset = last_y - y_pos;
                last_x               = x_pos;
                last_y               = y_pos;

                if (free_look_mode)
                {
                    // calculate offset

                    float sensitivity_scale = 1.0F;
                    if (camera_data.has_focus_point && camera_data.focus_constraint_enabled)
                    {
                        const float current_distance = glm::length(camera_data.position - camera_data.focus_point);
                        const float distance_factor  = glm::clamp(current_distance / camera_data.focus_distance,
                                                                 camera_data.min_focus_distance / camera_data.focus_distance,
                                                                 camera_data.max_focus_distance / camera_data.focus_distance);
                        sensitivity_scale            = distance_factor;
                    }
                    const float actual_x_offset = x_offset * camera_data.mouse_sensitivity * sensitivity_scale;
                    const float actual_y_offset = y_offset * camera_data.mouse_sensitivity * sensitivity_scale;

                    // offset yaw and pitch

                    camera_data.yaw += actual_x_offset;
                    camera_data.pitch += actual_y_offset;

                    // normalize camera transform

                    camera_data.pitch = std::min(camera_data.pitch, 89.0F);
                    camera_data.pitch = std::max(camera_data.pitch, -89.0F);

                    // update camera's uniform matrix

                    camera_data.update_camera_vectors();
                }

                if (camera_pan_mode)
                {
                    const float current_distance =
                        camera_data.has_focus_point ? glm::length(camera_data.position - camera_data.focus_point) : camera_data.focus_distance;
                    const float distance_scale           = glm::clamp(current_distance / camera_data.focus_distance,
                                                            camera_data.min_focus_distance / camera_data.focus_distance,
                                                            camera_data.max_focus_distance / camera_data.focus_distance);
                    constexpr float pan_speed_multiplier = 0.005F;
                    const float actual_pan_speed_multiplier =
                        camera_data.focus_constraint_enabled ? pan_speed_multiplier / distance_scale : pan_speed_multiplier;
                    const float target_x_offset = x_offset * camera_data.movement_speed * actual_pan_speed_multiplier;
                    const float target_y_offset = y_offset * camera_data.movement_speed * actual_pan_speed_multiplier;

                    camera_data.position -= camera_data.right * target_x_offset;
                    camera_data.position += camera_data.up * target_y_offset;
                }
            }
            break;

        case EventType::MouseWheel:
        {
            const float zoom_factor = camera_data.wheel_speed;
            const float distance    = glm::length(camera_data.position);

            if (event.mouse_wheel.y > 0)
            {
                if (distance > 0.5F)
                {
                    camera_data.position *= (1.0F - zoom_factor);
                }
            }
            else if (event.mouse_wheel.y < 0)
            {
                camera_data.position *= (1.0F + zoom_factor);
            }
            mouse_y_offset = event.mouse_wheel.y;
            process_mouse_input();
        }
        break;

        default:
            break;
        }
        process_keyboard_input();
    }

    glm::mat4 simple_camera::get_matrix(transform_matrix_type matrix_type) const
    {
        switch (matrix_type)
        {
        case transform_matrix_type::model:
            // default return identity matrix
            return {1.0F};
        case transform_matrix_type::view:
            return glm::lookAt(camera_data.position,
                               // camera position
                               camera_data.position + camera_data.front,
                               // camera looking at point
                               camera_data.up
                               // camera up direction
            );
        case transform_matrix_type::projection:
            return glm::perspective(glm::radians(camera_data.zoom),
                                    // FOV
                                    camera_data.aspect_ratio,
                                    // aspect ratio
                                    0.1F,
                                    // near plane
                                    1000.0F
                                    // far plane
            );
        default:
            return {1.0F};
        }
    }

    void simple_camera::process_keyboard_input()
    {
        // Calculate delta time
        auto current_tick_time = std::chrono::high_resolution_clock::now();
        float delta_time       = std::chrono::duration<float, std::chrono::seconds::period>(current_tick_time - last_tick_time).count();
        last_tick_time         = current_tick_time;

        // Prevent huge delta time if the camera hasn't been updated for a while (e.g. > 0.1s)
        // This ensures the first frame of movement doesn't jump too far
        if (delta_time > 0.1F)
            delta_time = 0.016F; // Assume 60fps for the first frame after idle

        // Check if any movement key is pressed
        bool is_moving = false;
        if (pressed_keys.contains(interface::KeyCode::W) || pressed_keys.contains(interface::KeyCode::Up) ||
            pressed_keys.contains(interface::KeyCode::A) || pressed_keys.contains(interface::KeyCode::Left) ||
            pressed_keys.contains(interface::KeyCode::S) || pressed_keys.contains(interface::KeyCode::Down) ||
            pressed_keys.contains(interface::KeyCode::D) || pressed_keys.contains(interface::KeyCode::Right) ||
            pressed_keys.contains(interface::KeyCode::Q) || pressed_keys.contains(interface::KeyCode::E))
        {
            is_moving = true;
        }

        if (is_moving)
        {
            current_speed_multiplier += acceleration_rate * delta_time;
            current_speed_multiplier = std::min(current_speed_multiplier, max_speed_multiplier);
        }
        else
        {
            current_speed_multiplier = 1.0F;
        }

        float velocity = camera_data.movement_speed * current_speed_multiplier;

        // If free look mode is enabled, move in world space based on camera
        // orientation
        if (free_look_mode)
        {
            // Calculate movement speed scale based on distance to focus point if
            // constraint is enabled
            float distance_scale = 1.0F;
            if (camera_data.has_focus_point && camera_data.focus_constraint_enabled)
            {
                float current_distance = glm::length(camera_data.position - camera_data.focus_point);
                distance_scale         = glm::clamp(current_distance / camera_data.focus_distance,
                                            camera_data.min_focus_distance / camera_data.focus_distance,
                                            camera_data.max_focus_distance / camera_data.focus_distance);
            }
            float current_velocity = velocity / distance_scale; // Slower when closer

            // move in the screen space
            glm::vec3 movement(0.0F);

            // move front/back (Z-axis relative to camera)
            if (pressed_keys.contains(interface::KeyCode::W) || pressed_keys.contains(interface::KeyCode::Up))
            {
                movement += camera_data.front * current_velocity;
            }
            if (pressed_keys.contains(interface::KeyCode::S) || pressed_keys.contains(interface::KeyCode::Down))
            {
                movement -= camera_data.front * current_velocity;
            }

            // move left/right (X-axis relative to camera)
            if (pressed_keys.contains(interface::KeyCode::A) || pressed_keys.contains(interface::KeyCode::Left))
            {
                movement -= camera_data.right * current_velocity;
            }
            if (pressed_keys.contains(interface::KeyCode::D) || pressed_keys.contains(interface::KeyCode::Right))
            {
                movement += camera_data.right * current_velocity;
            }

            // move up/down (Y-axis relative to world or camera up)
            if (pressed_keys.contains(interface::KeyCode::Q))
            {
                movement -= camera_data.up * current_velocity; // Using camera's up vector for local up/down
            }
            if (pressed_keys.contains(interface::KeyCode::E))
            {
                movement += camera_data.up * current_velocity; // Using camera's up vector for local up/down
            }

            // apply the movement
            camera_data.position += movement;
        }
        else // Original screen space movement logic
        {
            // move in the screen space
            glm::vec3 movement(0.0F);

            // move up (Y-axis)
            if (pressed_keys.contains(interface::KeyCode::W) || pressed_keys.contains(interface::KeyCode::Up))
            {
                movement.y += velocity; // move up (Y-axis positive direction)
            }
            if (pressed_keys.contains(interface::KeyCode::S) || pressed_keys.contains(interface::KeyCode::Down))
            {
                movement.y -= velocity; // move down (Y-axis negative direction)
            }

            // move left (l-axis)t (X-axis)
            if (pressed_keys.contains(interface::KeyCode::A) || pressed_keys.contains(interface::KeyCode::Left))
            {
                movement.x -= velocity; // move left (l-axis negative direction)X-axis
                // negative direction)
            }
            if (pressed_keys.contains(interface::KeyCode::D) || pressed_keys.contains(interface::KeyCode::Right))
            {
                movement.x += velocity; // move right (X-axis positive direction)
            }

            // move front (Z-axis)
            if (pressed_keys.contains(interface::KeyCode::Q))
            {
                movement.z += velocity; // move back (Z-axis negative direction)
            }
            if (pressed_keys.contains(interface::KeyCode::E))
            {
                movement.z -= velocity; // move front (Z-axis positive direction)
            }

            // apply the smooth movement (removed smooth factor for simplicity in free
            // look, could add back if needed) float smooth_factor = 0.1f; // smooth
            // factor
            camera_data.position += movement; // * smooth_factor;
        }
    }

    void simple_camera::process_mouse_input()
    {
        if (camera_data.has_focus_point && camera_data.focus_constraint_enabled)
        {
            // Zoom by moving along the camera's front vector
            float zoom_step = camera_data.movement_speed * 0.5F; // Adjust zoom speed

            // Calculate distance scale
            float current_distance = glm::length(camera_data.position - camera_data.focus_point);
            float distance_scale   = glm::clamp(current_distance / camera_data.focus_distance,
                                              camera_data.min_focus_distance / camera_data.focus_distance,
                                              camera_data.max_focus_distance / camera_data.focus_distance);
            zoom_step /= distance_scale; // Smaller steps when closer

            if (mouse_y_offset > 0)
            {
                // Zoom in (move towards focus point)
                camera_data.position += camera_data.front * zoom_step;
            }
            else if (mouse_y_offset < 0)
            {
                // Zoom out (move away from focus point)
                camera_data.position -= camera_data.front * zoom_step;
            }
            // Update camera vectors after changing position for orbit-like feel
            camera_data.update_camera_vectors();
        }
        else
        {
            // Original FOV zoom logic when focus constraint is disabled
            camera_data.zoom -= mouse_y_offset;
            camera_data.zoom = std::max(camera_data.zoom, 1.0F);
            camera_data.zoom = std::min(camera_data.zoom, 45.0F);
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
        case camera_attribute::front:
            return camera_data.front;
        case camera_attribute::up:
            return camera_data.up;
        case camera_attribute::right:
            return camera_data.right;
        case camera_attribute::yaw:
            return camera_data.yaw;
        case camera_attribute::pitch:
            return camera_data.pitch;
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
        case camera_attribute::zoom:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.zoom = *val;
            break;
        case camera_attribute::width:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.width = *val;
            break;
        case camera_attribute::aspect_ratio:
            if (const auto* val = std::any_cast<float>(&value))
                camera_data.aspect_ratio = *val;
            break;
        case camera_attribute::position:
            if (const auto* val = std::any_cast<glm::vec3>(&value))
                camera_data.position = *val;
            break;
        case camera_attribute::front:
            if (const auto* val = std::any_cast<glm::vec3>(&value))
                camera_data.front = *val;
            break;
        case camera_attribute::up:
            if (const auto* val = std::any_cast<glm::vec3>(&value))
                camera_data.up = *val;
            break;
        case camera_attribute::right:
            if (const auto* val = std::any_cast<glm::vec3>(&value))
                camera_data.right = *val;
            break;
        case camera_attribute::yaw:
            if (const auto* val = std::any_cast<float>(&value))
            {
                camera_data.yaw = *val;
                camera_data.update_camera_vectors();
            }
            break;
        case camera_attribute::pitch:
            if (const auto* val = std::any_cast<float>(&value))
            {
                camera_data.pitch = *val;
                camera_data.update_camera_vectors();
            }
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
        default:
            break;
        }
    }
}; // namespace interface
