#pragma once
#include <glm/glm.hpp>
#include <unordered_set>
#include <chrono>

#include "camera.h"
#include "input.h"

namespace interface
{


    class SimpleCamera : public Camera
    {
    public:
        SimpleCamera();
        ~SimpleCamera() override = default;

        void tick(const interface::InputEvent& event) override;
        glm::mat4 get_matrix(TransformMatrixType matrix_type) override;
        void set_attribute(CameraAttribute attribute, float value) override;

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
        
        // Time based acceleration
        std::chrono::high_resolution_clock::time_point last_tick_time_;
        float current_speed_multiplier_ = 1.0F;
        const float max_speed_multiplier_ = 20.0F; // Max 20x speed
        const float acceleration_rate_ = 10.0F;     // Increase 10x per second

        void process_keyboard_input() override;

        void process_mouse_scroll(float y_offset) override;
    };
} // namespace interface