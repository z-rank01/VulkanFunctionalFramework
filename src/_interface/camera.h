#pragma once
#include <glm/glm.hpp>

#include "input.h"

namespace interface
{
    enum class TransformMatrixType : std::uint8_t
    {
        kModel,
        kView,
        kProjection
    };

    enum class CameraAttribute : std::uint8_t
    {
        kMovementSpeed,
        kZoom,
        kWidth,
        kAspectRatio
    };

    struct SCamera
    {
        glm::vec3 position_{};
        glm::vec3 front_{};
        glm::vec3 up_{};
        glm::vec3 right_{};
        glm::vec3 world_up_;
        float yaw_;
        float pitch_;
        float movement_speed_{};
        float wheel_speed_{};
        float mouse_sensitivity_{};
        float zoom_{};
        float width_{};
        float aspect_ratio_{};

        // 聚焦点相关
        glm::vec3 focus_point_{};
        float focus_distance_{};
        float min_focus_distance_{};
        float max_focus_distance_{};
        bool has_focus_point_{};

        // Add focus constraint enabled flag
        bool focus_constraint_enabled_{};

        SCamera(glm::vec3 pos       = glm::vec3(0.0f, 0.0f, 0.0f),
                glm::vec3 up        = glm::vec3(0.0f, 1.0f, 0.0f),
                float initial_yaw   = -90.0f,
                float initial_pitch = 0.0f)
            : position_(pos), world_up_(up), yaw_(initial_yaw), pitch_(initial_pitch)
              // Initialize focus constraint enabled flag
               // Default to enabled
        {
            update_camera_vectors();
        }

        void update_camera_vectors()
        {
            // if (pitch > 89.0f) pitch = 89.0f;
            // if (pitch < -89.0f) pitch = -89.0f;

            // 在Vulkan坐标系中计算相机方向：+X向右，+Y向上，+Z向屏幕外
            glm::vec3 new_front;
            new_front.x = cos(glm::radians(yaw_)) * cos(glm::radians(pitch_));
            new_front.y = sin(glm::radians(pitch_)); // Y轴向上
            new_front.z = sin(glm::radians(yaw_)) * cos(glm::radians(pitch_));
            front_       = glm::normalize(new_front);

            // 计算右向量和上向量
            right_ = glm::normalize(glm::cross(front_, world_up_));
            up_    = glm::normalize(glm::cross(right_, front_));
        }
    };

    class Camera
    {
    public:
        virtual ~Camera() = default;

        // core loop functions per frame
        virtual void tick(const interface::InputEvent& event) = 0;
        // getter
        virtual glm::mat4 get_matrix(TransformMatrixType matrix_type) = 0;
        // setter
        virtual void set_attribute(CameraAttribute attribute, float value) = 0;

    private:
        // core helper functions
        virtual void process_keyboard_input() = 0;

        virtual void process_mouse_scroll(float y_offset) = 0;
    };
}