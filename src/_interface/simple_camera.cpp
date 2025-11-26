#include "simple_camera.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

namespace interface
{
    using glm::mat4;

    SimpleCamera::SimpleCamera()
    {
        // initialize camera attributes
        camera_.position_          = glm::vec3(0.0F, 0.0F, 10.0F); // 更远的初始距离
        camera_.yaw_               = -90.0F;                       // look at origin
        camera_.pitch_             = 0.0F;                         // horizontal view
        camera_.wheel_speed_       = 0.1F;                         // 降低滚轮速度，避免变化太剧烈
        camera_.movement_speed_    = 0.05F;                        // 调整移动速度
        camera_.mouse_sensitivity_ = 0.2F;                         // 降低鼠标灵敏度
        camera_.zoom_              = 45.0F;
        camera_.world_up_          = glm::vec3(0.0F, 1.0F, 0.0F); // Y-axis is up in Vulkan
        camera_.aspect_ratio_      = 16.0F / 9.0F;
        camera_.width_             = 1600;

        // initialize camera basic vectors
        camera_.front_ = glm::vec3(0.0F, 0.0F, -1.0F); // look at -z direction
        camera_.right_ = glm::vec3(1.0F, 0.0F, 0.0F);  // the right direction is +x
        camera_.up_    = glm::vec3(0.0F, 1.0F, 0.0F);  // up direction is +y (because Y-axis is up in Vulkan)

        // initialize focus point related parameters
        camera_.focus_point_        = glm::vec3(0.0F); // default focus on origin
        camera_.has_focus_point_    = true;            // default enables focus point
        camera_.focus_distance_     = 10.0F;           // 增加默认焦距
        camera_.min_focus_distance_ = 0.5F;            // minimum focus distance
        camera_.max_focus_distance_ = 10000.0F;        // maximum focus distance

        last_tick_time_ = std::chrono::high_resolution_clock::now();
    }

    void SimpleCamera::tick(const InputEvent& event)
    {
        switch (event.type)
        {
        case EventType::Quit:
            break;

        // keyboard event
        case EventType::KeyDown:
            pressed_keys_.insert(event.key.key);
            break;
        case EventType::KeyUp:
            pressed_keys_.erase(event.key.key);
            break;

        // mouse event
        case EventType::MouseButtonDown:
            last_x_          = event.mouse_button.x;
            last_y_          = event.mouse_button.y;
            free_look_mode_  = event.mouse_button.button == MouseButton::Right ? true : free_look_mode_;
            camera_pan_mode_ = event.mouse_button.button == MouseButton::Middle ? true : camera_pan_mode_;
            break;
        case EventType::MouseButtonUp:
            free_look_mode_  = event.mouse_button.button == MouseButton::Right ? false : free_look_mode_;
            camera_pan_mode_ = event.mouse_button.button == MouseButton::Middle ? false : camera_pan_mode_;
            break;
        case EventType::MouseMove:
            if (!(free_look_mode_ || camera_pan_mode_))
                break;
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
                    if (camera_.has_focus_point_ && camera_.focus_constraint_enabled_)
                    {
                        const float current_distance = glm::length(camera_.position_ - camera_.focus_point_);
                        const float distance_factor  = glm::clamp(current_distance / camera_.focus_distance_,
                                                                 camera_.min_focus_distance_ / camera_.focus_distance_,
                                                                 camera_.max_focus_distance_ / camera_.focus_distance_);
                        sensitivity_scale            = distance_factor;
                    }
                    const float actual_x_offset = x_offset * camera_.mouse_sensitivity_ * sensitivity_scale;
                    const float actual_y_offset = y_offset * camera_.mouse_sensitivity_ * sensitivity_scale;

                    // offset yaw and pitch
                    camera_.yaw_ += actual_x_offset;
                    camera_.pitch_ += actual_y_offset;

                    // normalize camera transform
                    camera_.pitch_ = std::min(camera_.pitch_, 89.0F);
                    camera_.pitch_ = std::max(camera_.pitch_, -89.0F);

                    // update camera's uniform matrix
                    camera_.update_camera_vectors();
                }

                if (camera_pan_mode_)
                {
                    const float current_distance =
                        camera_.has_focus_point_ ? glm::length(camera_.position_ - camera_.focus_point_) : camera_.focus_distance_;
                    const float distance_scale           = glm::clamp(current_distance / camera_.focus_distance_,
                                                            camera_.min_focus_distance_ / camera_.focus_distance_,
                                                            camera_.max_focus_distance_ / camera_.focus_distance_);
                    constexpr float pan_speed_multiplier = 0.005F;
                    const float actual_pan_speed_multiplier =
                        camera_.focus_constraint_enabled_ ? pan_speed_multiplier / distance_scale : pan_speed_multiplier;

                    const float target_x_offset = x_offset * camera_.movement_speed_ * actual_pan_speed_multiplier;
                    const float target_y_offset = y_offset * camera_.movement_speed_ * actual_pan_speed_multiplier;

                    camera_.position_ -= camera_.right_ * target_x_offset;
                    camera_.position_ += camera_.up_ * target_y_offset;
                }
            }
            break;

        case EventType::MouseWheel:
        {
            const float zoom_factor = camera_.wheel_speed_;
            const float distance    = glm::length(camera_.position_);

            if (event.mouse_wheel.y > 0)
            {
                if (distance > 0.5F)
                {
                    camera_.position_ *= (1.0F - zoom_factor);
                }
            }
            else if (event.mouse_wheel.y < 0)
            {
                camera_.position_ *= (1.0F + zoom_factor);
            }

            process_mouse_scroll(event.mouse_wheel.y);
        }
        break;

        default:
            break;
        }
        process_keyboard_input();
    }

    glm::mat4 SimpleCamera::get_matrix(TransformMatrixType matrix_type)
    {
        switch (matrix_type)
        {
        case TransformMatrixType::kModel:
            // default return identity matrix
            return {1.0F};
        case TransformMatrixType::kView:
            return glm::lookAt(camera_.position_,
                               // camera position
                               camera_.position_ + camera_.front_,
                               // camera looking at point
                               camera_.up_
                               // camera up direction
            );
        case TransformMatrixType::kProjection:
            return glm::perspective(glm::radians(camera_.zoom_),
                                    // FOV
                                    camera_.aspect_ratio_,
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

    void SimpleCamera::set_attribute(CameraAttribute attribute, float value)
    {
        switch (attribute)
        {
        case CameraAttribute::kMovementSpeed:
            camera_.movement_speed_ = value;
            break;
        case CameraAttribute::kZoom:
            camera_.zoom_ = value;
            break;
        case CameraAttribute::kWidth:
            camera_.width_ = value;
            break;
        case CameraAttribute::kAspectRatio:
            camera_.aspect_ratio_ = value;
            break;
        }
    }

    void SimpleCamera::process_keyboard_input()
    {
        // Calculate delta time
        auto current_tick_time = std::chrono::high_resolution_clock::now();
        float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(current_tick_time - last_tick_time_).count();
        last_tick_time_ = current_tick_time;

        // Prevent huge delta time if the camera hasn't been updated for a while (e.g. > 0.1s)
        // This ensures the first frame of movement doesn't jump too far
        if (delta_time > 0.1F) delta_time = 0.016F; // Assume 60fps for the first frame after idle

        // Check if any movement key is pressed
        bool is_moving = false;
        if (pressed_keys_.count(interface::KeyCode::W) || pressed_keys_.count(interface::KeyCode::Up) ||
            pressed_keys_.count(interface::KeyCode::S) || pressed_keys_.count(interface::KeyCode::Down) ||
            pressed_keys_.count(interface::KeyCode::A) || pressed_keys_.count(interface::KeyCode::Left) ||
            pressed_keys_.count(interface::KeyCode::D) || pressed_keys_.count(interface::KeyCode::Right) ||
            pressed_keys_.count(interface::KeyCode::Q) || pressed_keys_.count(interface::KeyCode::E))
        {
            is_moving = true;
        }

        if (is_moving)
        {
            current_speed_multiplier_ += acceleration_rate_ * delta_time;
            if (current_speed_multiplier_ > max_speed_multiplier_)
            {
                current_speed_multiplier_ = max_speed_multiplier_;
            }
        }
        else
        {
            current_speed_multiplier_ = 1.0F;
        }

        float velocity = camera_.movement_speed_ * current_speed_multiplier_;

        // If free look mode is enabled, move in world space based on camera
        // orientation
        if (free_look_mode_)
        {
            // Calculate movement speed scale based on distance to focus point if
            // constraint is enabled
            float distance_scale = 1.0F;
            if (camera_.has_focus_point_ && camera_.focus_constraint_enabled_)
            {
                float current_distance = glm::length(camera_.position_ - camera_.focus_point_);
                distance_scale         = glm::clamp(current_distance / camera_.focus_distance_,
                                            camera_.min_focus_distance_ / camera_.focus_distance_,
                                            camera_.max_focus_distance_ / camera_.focus_distance_);
            }
            float current_velocity = velocity / distance_scale; // Slower when closer

            // move in the screen space
            glm::vec3 movement(0.0F);

            // move front/back (Z-axis relative to camera)
            if (pressed_keys_.count(interface::KeyCode::W) || pressed_keys_.count(interface::KeyCode::Up))
            {
                movement += camera_.front_ * current_velocity;
            }
            if (pressed_keys_.count(interface::KeyCode::S) || pressed_keys_.count(interface::KeyCode::Down))
            {
                movement -= camera_.front_ * current_velocity;
            }

            // move left/right (X-axis relative to camera)
            if (pressed_keys_.count(interface::KeyCode::A) || pressed_keys_.count(interface::KeyCode::Left))
            {
                movement -= camera_.right_ * current_velocity;
            }
            if (pressed_keys_.count(interface::KeyCode::D) || pressed_keys_.count(interface::KeyCode::Right))
            {
                movement += camera_.right_ * current_velocity;
            }

            // move up/down (Y-axis relative to world or camera up)
            if (pressed_keys_.count(interface::KeyCode::Q))
            {
                movement -= camera_.up_ * current_velocity; // Using camera's up vector for local up/down
            }
            if (pressed_keys_.count(interface::KeyCode::E))
            {
                movement += camera_.up_ * current_velocity; // Using camera's up vector for local up/down
            }

            // apply the movement
            camera_.position_ += movement;
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
            camera_.position_ += movement; // * smooth_factor;
        }
    }

    void SimpleCamera::process_mouse_scroll(float y_offset)
    {
        if (camera_.has_focus_point_ && camera_.focus_constraint_enabled_)
        {
            // Zoom by moving along the camera's front vector
            float zoom_step = camera_.movement_speed_ * 0.5F; // Adjust zoom speed

            // Calculate distance scale
            float current_distance = glm::length(camera_.position_ - camera_.focus_point_);
            float distance_scale   = glm::clamp(current_distance / camera_.focus_distance_,
                                              camera_.min_focus_distance_ / camera_.focus_distance_,
                                              camera_.max_focus_distance_ / camera_.focus_distance_);
            zoom_step /= distance_scale; // Smaller steps when closer

            if (y_offset > 0)
            {
                // Zoom in (move towards focus point)
                camera_.position_ += camera_.front_ * zoom_step;
            }
            else if (y_offset < 0)
            {
                // Zoom out (move away from focus point)
                camera_.position_ -= camera_.front_ * zoom_step;
            }
            // Update camera vectors after changing position for orbit-like feel
            camera_.update_camera_vectors();
        }
        else
        {
            // Original FOV zoom logic when focus constraint is disabled
            camera_.zoom_ -= y_offset;
            camera_.zoom_ = std::max(camera_.zoom_, 1.0F);
            camera_.zoom_ = std::min(camera_.zoom_, 45.0F);
        }
    }
}; // namespace interface