#pragma once
#include <unordered_set>

#include "input.h"

namespace interface
{
    class Camera
    {
    public:
        virtual ~Camera() = default;

        // core loop functions per frame
        virtual void tick(const interface::InputEvent& event) = 0;

    private:
        // core helper functions
        virtual void process_keyboard_input(float delta_time) = 0;

        virtual void process_mouse_scroll(float y_offset) = 0;
    };
}