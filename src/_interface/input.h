#pragma once

#include <cstdint>

namespace interface
{

    enum class KeyCode
    {
        Unknown,
        W,
        A,
        S,
        D,
        Q,
        E,
        F,
        Space,
        LShift,
        LCtrl,
        Escape,
        Up,
        Down,
        Left,
        Right
    };

    enum class MouseButton
    {
        Left,
        Middle,
        Right,
        Unknown
    };

    enum class EventType
    {
        None, 
        Quit,
        Resize,
        KeyDown,
        KeyUp,
        MouseMove,
        MouseWheel,
        MouseButtonDown,
        MouseButtonUp
    };

    struct ResizeEvent
    {
        int width;
        int height;
    };

    struct KeyEvent
    {
        KeyCode key;
    };

    struct MouseMoveEvent
    {
        float x;
        float y;
        float xrel;
        float yrel;
    };

    struct MouseWheelEvent
    {
        float x;
        float y;
    };

    struct MouseButtonEvent
    {
        MouseButton button;
        float x;
        float y;
        bool pressed; // true for down, false for up
    };

    /**
     * @struct InputEvent
     *
     * @brief Represents a generic input event used in window implementation and camera implementation.
     *
     * This structure serves as a unified container for various types of
     * input events, such as keyboard, mouse, window resize, and quit events.
     * The specific type of event is determined by the `type` member, which uses
     * the EventType enumeration.
     *
     * The union within the structure allows efficient storage of data for
     * the specific type of event being handled.
     *
     * @note Only one member of the union is valid at any given time,
     * depending on the value of the `type` field.
     *
     * @details
     * The fields in the union are as follows:
     * - `resize`: Contains information about a window resize event.
     * - `key`: Contains information about a keyboard event, such as KeyDown or KeyUp.
     * - `mouse_move`: Holds details for a mouse movement event.
     * - `mouse_wheel`: Represents information about mouse wheel scrolling.
     * - `mouse_button`: Contains data for mouse button press or release events.
     */
    struct InputEvent
    {
        EventType type;

        union
        {
            ResizeEvent resize;
            KeyEvent key;
            MouseMoveEvent mouse_move;
            MouseWheelEvent mouse_wheel;
            MouseButtonEvent mouse_button;
        };
    };

} // namespace interface