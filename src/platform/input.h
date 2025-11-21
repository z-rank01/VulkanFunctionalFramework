#pragma once

#include <cstdint>

namespace platform {

enum class KeyCode {
    Unknown,
    W, A, S, D, Q, E, F,
    Space, LShift, LCtrl, Escape,
    Up, Down, Left, Right
};

enum class MouseButton {
    Left, Middle, Right, Unknown
};

enum class EventType {
    Quit,
    Resize,
    KeyDown,
    KeyUp,
    MouseMove,
    MouseWheel,
    MouseButtonDown,
    MouseButtonUp
};

struct ResizeEvent {
    int width;
    int height;
};

struct KeyEvent {
    KeyCode key;
};

struct MouseMoveEvent {
    float x;
    float y;
    float xrel;
    float yrel;
};

struct MouseWheelEvent {
    float x;
    float y;
};

struct MouseButtonEvent {
    MouseButton button;
    float x;
    float y;
    bool pressed; // true for down, false for up
};

struct InputEvent {
    EventType type;
    union {
        ResizeEvent resize;
        KeyEvent key;
        MouseMoveEvent mouse_move;
        MouseWheelEvent mouse_wheel;
        MouseButtonEvent mouse_button;
    };
};

} // namespace platform
