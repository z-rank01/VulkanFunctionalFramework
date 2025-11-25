#include "simple_camera.h"

namespace interface
{

    void SimpleCamera::tick(const interface::InputEvent& event)
    {
        switch (event.type)
        {
        case interface::EventType::Quit:
            break;

        // keyboard event
        case interface::EventType::KeyDown:
            pressed_keys_.insert(event.key.key);
            break;
        case interface::EventType::KeyUp:
            pressed_keys_.erase(event.key.key);
            break;

        // mouse event
        case interface::EventType::MouseButtonDown:
            last_x_ = event.mouse_button.x;
            last_y_ = event.mouse_button.y;
            free_look_mode_ = event.mouse_button.button == interface::MouseButton::Right ? true : free_look_mode_;
            camera_pan_mode_ = event.mouse_button.button == interface::MouseButton::Middle ? true : camera_pan_mode_;
            break;
        case interface::EventType::MouseButtonUp:
            free_look_mode_ = event.mouse_button.button == interface::MouseButton::Right ? false : free_look_mode_;
            camera_pan_mode_ = event.mouse_button.button == interface::MouseButton::Middle ? false : camera_pan_mode_;
            break;
        case interface::EventType::MouseMove:
            if (!(free_look_mode_ || camera_pan_mode_)) break;

            {
                const float x_pos    = event.mouse_move.x;
                const float y_pos    = event.mouse_move.y;
                const float x_offset = x_pos - last_x_;
                const float y_offset = last_y_ - y_pos;
                last_x_              = x_pos;
                last_y_              = y_pos;

                if (free_look_mode_)
                {
                    // calculate offset
                    float sensitivity_scale = 1.0F;
                    if (camera_.has_focus_point && camera_.focus_constraint_enabled_)
                    {
                        const float current_distance = glm::length(camera_.position - camera_.focus_point);
                        const float distance_factor  = glm::clamp(current_distance / camera_.focus_distance,
                                                                 camera_.min_focus_distance / camera_.focus_distance,
                                                                 camera_.max_focus_distance / camera_.focus_distance);
                        sensitivity_scale = distance_factor;
                    }
                    const float actual_x_offset = x_offset * camera_.mouse_sensitivity * sensitivity_scale;
                    const float actual_y_offset = y_offset * camera_.mouse_sensitivity * sensitivity_scale;

                    // offset yaw and pitch
                    camera_.yaw += actual_x_offset;
                    camera_.pitch += actual_y_offset;

                    // normalize camera transform
                    camera_.pitch = std::min(camera_.pitch, 89.0F);
                    camera_.pitch = std::max(camera_.pitch, -89.0F);

                    // update camera's uniform matrix
                    camera_.UpdateCameraVectors();
                }

                if (camera_pan_mode_)
                {
                    const float current_distance = camera_.has_focus_point
                                                       ? glm::length(camera_.position - camera_.focus_point)
                                                       : camera_.focus_distance;
                    const float distance_scale = glm::clamp(current_distance / camera_.focus_distance,
                                                            camera_.min_focus_distance / camera_.focus_distance,
                                                            camera_.max_focus_distance / camera_.focus_distance);
                    constexpr float pan_speed_multiplier    = 0.005F;
                    const float actual_pan_speed_multiplier = camera_.focus_constraint_enabled_
                                                                  ? pan_speed_multiplier / distance_scale
                                                                  : pan_speed_multiplier;

                    const float target_x_offset = x_offset * camera_.movement_speed * actual_pan_speed_multiplier;
                    const float target_y_offset = y_offset * camera_.movement_speed * actual_pan_speed_multiplier;

                    camera_.position -= camera_.right * target_x_offset;
                    camera_.position += camera_.up * target_y_offset;
                }
            }
            break;

        case interface::EventType::MouseWheel:
        {
            const float zoom_factor = camera_.wheel_speed;
            const float distance    = glm::length(camera_.position);

            if (event.mouse_wheel.y > 0)
            {
                if (distance > 0.5F)
                {
                    camera_.position *= (1.0F - zoom_factor);
                }
            }
            else if (event.mouse_wheel.y < 0)
            {
                camera_.position *= (1.0F + zoom_factor);
            }

            process_mouse_scroll(event.mouse_wheel.y);
        }
        break;

        default:
            break;
        }
    }

    void SimpleCamera::process_keyboard_input(float delta_time)
    {
        float velocity = camera_.movement_speed * delta_time;

        // If free look mode is enabled, move in world space based on camera
        // orientation
        if (free_look_mode_)
        {
            // Calculate movement speed scale based on distance to focus point if
            // constraint is enabled
            float distance_scale = 1.0F;
            if (camera_.has_focus_point && camera_.focus_constraint_enabled_)
            {
                float current_distance = glm::length(camera_.position - camera_.focus_point);
                distance_scale         = glm::clamp(current_distance / camera_.focus_distance,
                                            camera_.min_focus_distance / camera_.focus_distance,
                                            camera_.max_focus_distance / camera_.focus_distance);
            }
            float current_velocity = velocity / distance_scale; // Slower when closer

            // move in the screen space
            glm::vec3 movement(0.0F);

            // move front/back (Z-axis relative to camera)
            if (pressed_keys_.count(interface::KeyCode::W) || pressed_keys_.count(interface::KeyCode::Up))
            {
                movement += camera_.front * current_velocity;
            }
            if (pressed_keys_.count(interface::KeyCode::S) || pressed_keys_.count(interface::KeyCode::Down))
            {
                movement -= camera_.front * current_velocity;
            }

            // move left/right (X-axis relative to camera)
            if (pressed_keys_.count(interface::KeyCode::A) || pressed_keys_.count(interface::KeyCode::Left))
            {
                movement -= camera_.right * current_velocity;
            }
            if (pressed_keys_.count(interface::KeyCode::D) || pressed_keys_.count(interface::KeyCode::Right))
            {
                movement += camera_.right * current_velocity;
            }

            // move up/down (Y-axis relative to world or camera up)
            if (pressed_keys_.count(interface::KeyCode::Q))
            {
                movement -= camera_.up * current_velocity; // Using camera's up vector for local up/down
            }
            if (pressed_keys_.count(interface::KeyCode::E))
            {
                movement += camera_.up * current_velocity; // Using camera's up vector for local up/down
            }

            // apply the movement
            camera_.position += movement;
        }
        else // Original screen space movement logic
        {
            // move in the screen space
            glm::vec3 movement(0.0F);

            // move up (Y-axis)
            if (pressed_keys_.count(interface::KeyCode::W) || pressed_keys_.count(interface::KeyCode::Up))
            {
                movement.y += velocity; // move up (Y-axis positive direction)
            }
            if (pressed_keys_.count(interface::KeyCode::S) || pressed_keys_.count(interface::KeyCode::Down))
            {
                movement.y -= velocity; // move down (Y-axis negative direction)
            }

            // move left (l-axis)t (X-axis)
            if (pressed_keys_.count(interface::KeyCode::A) || pressed_keys_.count(interface::KeyCode::Left))
            {
                movement.x -= velocity; // move left (l-axis negative direction)X-axis
                // negative direction)
            }
            if (pressed_keys_.count(interface::KeyCode::D) || pressed_keys_.count(interface::KeyCode::Right))
            {
                movement.x += velocity; // move right (X-axis positive direction)
            }

            // move front (Z-axis)
            if (pressed_keys_.count(interface::KeyCode::Q))
            {
                movement.z += velocity; // move back (Z-axis negative direction)
            }
            if (pressed_keys_.count(interface::KeyCode::E))
            {
                movement.z -= velocity; // move front (Z-axis positive direction)
            }

            // apply the smooth movement (removed smooth factor for simplicity in free
            // look, could add back if needed) float smooth_factor = 0.1f; // smooth
            // factor
            camera_.position += movement; // * smooth_factor;
        }
    }

    void SimpleCamera::process_mouse_scroll(float yoffset)
    {
        if (camera_.has_focus_point && camera_.focus_constraint_enabled_)
        {
            // Zoom by moving along the camera's front vector
            float zoom_step = camera_.movement_speed * 0.5F; // Adjust zoom speed

            // Calculate distance scale
            float current_distance = glm::length(camera_.position - camera_.focus_point);
            float distance_scale   = glm::clamp(current_distance / camera_.focus_distance,
                                              camera_.min_focus_distance / camera_.focus_distance,
                                              camera_.max_focus_distance / camera_.focus_distance);
            zoom_step /= distance_scale; // Smaller steps when closer

            if (yoffset > 0)
            {
                // Zoom in (move towards focus point)
                camera_.position += camera_.front * zoom_step;
            }
            else if (yoffset < 0)
            {
                // Zoom out (move away from focus point)
                camera_.position -= camera_.front * zoom_step;
            }
            // Update camera vectors after changing position for orbit-like feel
            camera_.UpdateCameraVectors();
        }
        else
        {
            // Original FOV zoom logic when focus constraint is disabled
            camera_.zoom -= yoffset;
            camera_.zoom = std::max(camera_.zoom, 1.0F);
            camera_.zoom = std::min(camera_.zoom, 45.0F);
        }
    }
};