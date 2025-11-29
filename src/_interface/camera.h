#pragma once
#include <any>
#include <glm/glm.hpp>
#include <stdexcept>

#include "input.h"

namespace interface
{
    enum class transform_matrix_type : std::uint8_t
    {
        model,
        view,
        projection
    };

    enum class camera_attribute : std::uint8_t
    {
        movement_speed,
        zoom,
        width,
        aspect_ratio,
        position,
        front,
        up,
        right,
        yaw,
        pitch,
        mouse_sensitivity,
        wheel_speed,
        focus_point,
        focus_distance,
        has_focus_point,
        focus_constraint_enabled
    };

    struct camera_data
    {
        glm::vec3 position{};
        glm::vec3 front{};
        glm::vec3 up{};
        glm::vec3 right{};
        glm::vec3 world_up;
        float yaw;
        float pitch;
        float movement_speed{};
        float wheel_speed{};
        float mouse_sensitivity{};
        float zoom{};
        float width{};
        float aspect_ratio{};

        // 聚焦点相关
        glm::vec3 focus_point{};
        float focus_distance{};
        float min_focus_distance{};
        float max_focus_distance{};
        bool has_focus_point{};

        // Add focus constraint enabled flag
        bool focus_constraint_enabled{};

        camera_data(glm::vec3 pos = glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float initial_yaw = -90.0f, float initial_pitch = 0.0f) : position(pos), world_up(up), yaw(initial_yaw), pitch(initial_pitch)
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
            new_front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
            new_front.y = sin(glm::radians(pitch)); // Y轴向上
            new_front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
            front       = glm::normalize(new_front);

            // 计算右向量和上向量
            right = glm::normalize(glm::cross(front, world_up));
            up    = glm::normalize(glm::cross(right, front));
        }
    };

    class camera
    {
    public:
        virtual ~camera() = default;

        // core loop functions per frame
        virtual void tick(const interface::InputEvent& event) = 0;

        // getter - returns type based on attribute
        template <typename T>
        T get(camera_attribute attribute) const
        {
            return get_impl<T>(attribute);
        }

        // setter - accepts any type based on attribute
        template <typename T>
        void set(camera_attribute attribute, const T& value)
        {
            set_impl<T>(attribute, value);
        }

        // special getter - returns matrix with specific space(model, view, projection)
        virtual glm::mat4 get_matrix(transform_matrix_type matrix_type) const = 0;

    private:
        // Internal implementation for get/set - derived classes override these
        // Uses std::any for type-safe value passing without const_cast

        virtual std::any get_impl_internal(camera_attribute attribute) const              = 0;
        virtual void set_impl_internal(camera_attribute attribute, const std::any& value) = 0;

        // core helper functions for input processing

        virtual void process_keyboard_input() = 0;
        virtual void process_mouse_input()    = 0;

        // Template method implementations (non-virtual, call virtual impl)

        template <typename T>
        T get_impl(camera_attribute attribute) const
        {
            std::any result = get_impl_internal(attribute);
            if (const T* ptr = std::any_cast<T>(&result))
            {
                return *ptr;
            }
            throw std::bad_any_cast();
        }

        template <typename T>
        void set_impl(camera_attribute attribute, const T& value)
        {
            set_impl_internal(attribute, std::any(value));
        }
    };
} // namespace interface
