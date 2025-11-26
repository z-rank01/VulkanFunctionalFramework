#include "simple_camera.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

namespace interface
{
    using glm::mat4;

    simple_camera::simple_camera()
    {
        // initialize camera attributes
        camera.position          = glm::vec3(0.0F, 0.0F, 10.0F); // 更远的初始距离
        camera.yaw               = -90.0F;                       // look at origin
        camera.pitch             = 0.0F;                         // horizontal view
        camera.wheel_speed       = 0.1F;                         // 降低滚轮速度，避免变化太剧烈
        camera.movement_speed    = 0.05F;                        // 调整移动速度
        camera.mouse_sensitivity = 0.2F;                         // 降低鼠标灵敏度
        camera.zoom              = 45.0F;
        camera.world_up          = glm::vec3(0.0F, 1.0F, 0.0F); // Y-axis is up in Vulkan
        camera.aspect_ratio      = 16.0F / 9.0F;
        camera.width             = 1600;

        // initialize camera basic vectors
        camera.front = glm::vec3(0.0F, 0.0F, -1.0F); // look at -z direction
        camera.right = glm::vec3(1.0F, 0.0F, 0.0F);  // the right direction is +x
        camera.up    = glm::vec3(0.0F, 1.0F, 0.0F);  // up direction is +y (because Y-axis is up in Vulkan)

        // initialize focus point related parameters
        camera.focus_point        = glm::vec3(0.0F); // default focus on origin
        camera.has_focus_point    = true;            // default enables focus point
        camera.focus_distance     = 10.0F;           // 增加默认焦距
        camera.min_focus_distance = 0.5F;            // minimum focus distance
        camera.max_focus_distance = 10000.0F;        // maximum focus distance

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
                last_x              = x_pos;
                last_y              = y_pos;

                if (free_look_mode)
                {
                    // calculate offset
                    float sensitivity_scale = 1.0F;
                    if (camera.has_focus_point && camera.focus_constraint_enabled)
                    {
                        const float current_distance = glm::length(camera.position - camera.focus_point);
                        const float distance_factor  = glm::clamp(current_distance / camera.focus_distance,
                                                                 camera.min_focus_distance / camera.focus_distance,
                                                                 camera.max_focus_distance / camera.focus_distance);
                        sensitivity_scale            = distance_factor;
                    }
                    const float actual_x_offset = x_offset * camera.mouse_sensitivity * sensitivity_scale;
                    const float actual_y_offset = y_offset * camera.mouse_sensitivity * sensitivity_scale;

                    // offset yaw and pitch
                    camera.yaw += actual_x_offset;
                    camera.pitch += actual_y_offset;

                    // normalize camera transform
                    camera.pitch = std::min(camera.pitch, 89.0F);
                    camera.pitch = std::max(camera.pitch, -89.0F);

                    // update camera's uniform matrix
                    camera.update_camera_vectors();
                }

                if (camera_pan_mode)
                {
                    const float current_distance =
                        camera.has_focus_point ? glm::length(camera.position - camera.focus_point) : camera.focus_distance;
                    const float distance_scale           = glm::clamp(current_distance / camera.focus_distance,
                                                            camera.min_focus_distance / camera.focus_distance,
                                                            camera.max_focus_distance / camera.focus_distance);
                    constexpr float pan_speed_multiplier = 0.005F;
                    const float actual_pan_speed_multiplier =
                        camera.focus_constraint_enabled ? pan_speed_multiplier / distance_scale : pan_speed_multiplier;

                    const float target_x_offset = x_offset * camera.movement_speed * actual_pan_speed_multiplier;
                    const float target_y_offset = y_offset * camera.movement_speed * actual_pan_speed_multiplier;

                    camera.position -= camera.right * target_x_offset;
                    camera.position += camera.up * target_y_offset;
                }
            }
            break;

        case EventType::MouseWheel:
        {
            const float zoom_factor = camera.wheel_speed;
            const float distance    = glm::length(camera.position);

            if (event.mouse_wheel.y > 0)
            {
                if (distance > 0.5F)
                {
                    camera.position *= (1.0F - zoom_factor);
                }
            }
            else if (event.mouse_wheel.y < 0)
            {
                camera.position *= (1.0F + zoom_factor);
            }

            process_mouse_scroll(event.mouse_wheel.y);
        }
        break;

        default:
            break;
        }
        process_keyboard_input();
    }

    glm::mat4 simple_camera::get_matrix(transform_matrix_type matrix_type)
    {
        switch (matrix_type)
        {
        case transform_matrix_type::model:
            // default return identity matrix
            return {1.0F};
        case transform_matrix_type::view:
            return glm::lookAt(camera.position,
                               // camera position
                               camera.position + camera.front,
                               // camera looking at point
                               camera.up
                               // camera up direction
            );
        case transform_matrix_type::projection:
            return glm::perspective(glm::radians(camera.zoom),
                                    // FOV
                                    camera.aspect_ratio,
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

    void simple_camera::set_attribute(camera_attribute attribute, float value)
    {
        switch (attribute)
        {
        case camera_attribute::movement_speed:
            camera.movement_speed = value;
            break;
        case camera_attribute::zoom:
            camera.zoom = value;
            break;
        case camera_attribute::width:
            camera.width = value;
            break;
        case camera_attribute::aspect_ratio:
            camera.aspect_ratio = value;
            break;
        }
    }

    void simple_camera::process_keyboard_input()
    {
        // Calculate delta time
        auto current_tick_time = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(current_tick_time - last_tick_time).count();
        last_tick_time = current_tick_time;

        // Prevent huge delta time if the camera hasn't been updated for a while (e.g. > 0.1s)
        // This ensures the first frame of movement doesn't jump too far
        if (delta_time > 0.1F) delta_time = 0.016F; // Assume 60fps for the first frame after idle

        // Check if any movement key is pressed
        bool is_moving = false;
        if (pressed_keys.count(interface::KeyCode::W) || pressed_keys.count(interface::KeyCode::Up) ||
            pressed_keys.count(interface::KeyCode::S) || pressed_keys.count(interface::KeyCode::Down) ||
            pressed_keys.count(interface::KeyCode::A) || pressed_keys.count(interface::KeyCode::Left) ||
            pressed_keys.count(interface::KeyCode::D) || pressed_keys.count(interface::KeyCode::Right) ||
            pressed_keys.count(interface::KeyCode::Q) || pressed_keys.count(interface::KeyCode::E))
        {
            is_moving = true;
        }

        if (is_moving)
        {
            current_speed_multiplier += acceleration_rate * delta_time;
            if (current_speed_multiplier > max_speed_multiplier)
            {
                current_speed_multiplier = max_speed_multiplier;
            }
        }
        else
        {
            current_speed_multiplier = 1.0F;
        }

        float velocity = camera.movement_speed * current_speed_multiplier;

        // If free look mode is enabled, move in world space based on camera
        // orientation
        if (free_look_mode)
        {
            // Calculate movement speed scale based on distance to focus point if
            // constraint is enabled
            float distance_scale = 1.0F;
            if (camera.has_focus_point && camera.focus_constraint_enabled)
            {
                float current_distance = glm::length(camera.position - camera.focus_point);
                distance_scale         = glm::clamp(current_distance / camera.focus_distance,
                                            camera.min_focus_distance / camera.focus_distance,
                                            camera.max_focus_distance / camera.focus_distance);
            }
            float current_velocity = velocity / distance_scale; // Slower when closer

            // move in the screen space
            glm::vec3 movement(0.0F);

            // move front/back (Z-axis relative to camera)
            if (pressed_keys.count(interface::KeyCode::W) || pressed_keys.count(interface::KeyCode::Up))
            {
                movement += camera.front * current_velocity;
            }
            if (pressed_keys.count(interface::KeyCode::S) || pressed_keys.count(interface::KeyCode::Down))
            {
                movement -= camera.front * current_velocity;
            }

            // move left/right (X-axis relative to camera)
            if (pressed_keys.count(interface::KeyCode::A) || pressed_keys.count(interface::KeyCode::Left))
            {
                movement -= camera.right * current_velocity;
            }
            if (pressed_keys.count(interface::KeyCode::D) || pressed_keys.count(interface::KeyCode::Right))
            {
                movement += camera.right * current_velocity;
            }

            // move up/down (Y-axis relative to world or camera up)
            if (pressed_keys.count(interface::KeyCode::Q))
            {
                movement -= camera.up * current_velocity; // Using camera's up vector for local up/down
            }
            if (pressed_keys.count(interface::KeyCode::E))
            {
                movement += camera.up * current_velocity; // Using camera's up vector for local up/down
            }

            // apply the movement
            camera.position += movement;
        }
        else // Original screen space movement logic
        {
            // move in the screen space
            glm::vec3 movement(0.0F);

            // move up (Y-axis)
            if (pressed_keys.count(interface::KeyCode::W) || pressed_keys.count(interface::KeyCode::Up))
            {
                movement.y += velocity; // move up (Y-axis positive direction)
            }
            if (pressed_keys.count(interface::KeyCode::S) || pressed_keys.count(interface::KeyCode::Down))
            {
                movement.y -= velocity; // move down (Y-axis negative direction)
            }

            // move left (l-axis)t (X-axis)
            if (pressed_keys.count(interface::KeyCode::A) || pressed_keys.count(interface::KeyCode::Left))
            {
                movement.x -= velocity; // move left (l-axis negative direction)X-axis
                // negative direction)
            }
            if (pressed_keys.count(interface::KeyCode::D) || pressed_keys.count(interface::KeyCode::Right))
            {
                movement.x += velocity; // move right (X-axis positive direction)
            }

            // move front (Z-axis)
            if (pressed_keys.count(interface::KeyCode::Q))
            {
                movement.z += velocity; // move back (Z-axis negative direction)
            }
            if (pressed_keys.count(interface::KeyCode::E))
            {
                movement.z -= velocity; // move front (Z-axis positive direction)
            }

            // apply the smooth movement (removed smooth factor for simplicity in free
            // look, could add back if needed) float smooth_factor = 0.1f; // smooth
            // factor
            camera.position += movement; // * smooth_factor;
        }
    }

    void simple_camera::process_mouse_scroll(float y_offset)
    {
        if (camera.has_focus_point && camera.focus_constraint_enabled)
        {
            // Zoom by moving along the camera's front vector
            float zoom_step = camera.movement_speed * 0.5F; // Adjust zoom speed

            // Calculate distance scale
            float current_distance = glm::length(camera.position - camera.focus_point);
            float distance_scale   = glm::clamp(current_distance / camera.focus_distance,
                                              camera.min_focus_distance / camera.focus_distance,
                                              camera.max_focus_distance / camera.focus_distance);
            zoom_step /= distance_scale; // Smaller steps when closer

            if (y_offset > 0)
            {
                // Zoom in (move towards focus point)
                camera.position += camera.front * zoom_step;
            }
            else if (y_offset < 0)
            {
                // Zoom out (move away from focus point)
                camera.position -= camera.front * zoom_step;
            }
            // Update camera vectors after changing position for orbit-like feel
            camera.update_camera_vectors();
        }
        else
        {
            // Original FOV zoom logic when focus constraint is disabled
            camera.zoom -= y_offset;
            camera.zoom = std::max(camera.zoom, 1.0F);
            camera.zoom = std::min(camera.zoom, 45.0F);
        }
    }
}; // namespace interface