#pragma once
#include <glm/glm.hpp>

#include "camera.h"
#include "input.h"

namespace interface
{
    struct SCamera
    {
        glm::vec3 position;
        glm::vec3 front;
        glm::vec3 up;
        glm::vec3 right;
        glm::vec3 world_up;
        float yaw;
        float pitch;
        float movement_speed;
        float wheel_speed;
        float mouse_sensitivity;
        float zoom;

        // 聚焦点相关
        glm::vec3 focus_point;
        bool has_focus_point;
        float focus_distance;
        float min_focus_distance;
        float max_focus_distance;

        // Add focus constraint enabled flag
        bool focus_constraint_enabled_;

        SCamera(glm::vec3 pos       = glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3 up        = glm::vec3(0.0f, 1.0f, 0.0f),
                float initial_yaw   = -90.0f,
                float initial_pitch = 0.0f)
            : position(pos), world_up(up), yaw(initial_yaw), pitch(initial_pitch), movement_speed(2.5f),
              wheel_speed(0.01f),
              mouse_sensitivity(0.1f), zoom(45.0f),
              // Initialize focus constraint enabled flag
              focus_constraint_enabled_(true) // Default to enabled
        {
            UpdateCameraVectors();
        }

        void UpdateCameraVectors()
        {
            // if (pitch > 89.0f) pitch = 89.0f;
            // if (pitch < -89.0f) pitch = -89.0f;

            // 在Vulkan坐标系中计算相机方向：+X向右，+Y向上，+Z向屏幕外
            glm::vec3 new_front;
            new_front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
            new_front.y = sin(glm::radians(pitch)); // Y轴向上
            new_front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
            front       = glm::normalize(new_front);

            // 计算右向量和上向量
            right = glm::normalize(glm::cross(front, world_up));
            up    = glm::normalize(glm::cross(right, front));
        }
    };

    class SimpleCamera : Camera
    {
    public:
        SimpleCamera() = default;
        ~SimpleCamera() override {}
        void tick(const interface::InputEvent& event) override;

    private:
        SCamera camera_;

        // Input handling members
        std::unordered_set<interface::KeyCode> pressed_keys_;
        float last_x_ = 0.0F;
        float last_y_ = 0.0F;
        // Rename camera_rotation_mode_ to free_look_mode_
        bool free_look_mode_  = false; // 相机自由查看模式标志（右键按住）
        bool camera_pan_mode_ = false; // 相机平移模式标志（中键）
        float orbit_distance_ = 0.0F;  // 轨道旋转时与中心的距离

        void process_keyboard_input(float delta_time) override;

        void process_mouse_scroll(float y_offset) override;
    };
} // namespace interface