#include <memory>

#include "_interface/camera.h"
#include "_interface/window.h"

struct window_config
{
    int width;
    int height;
    std::string title;
    [[nodiscard]] constexpr auto validate() const -> bool { return width > 0 && height > 0; }
};


class app_sample
{
public:
    // core public function
    void initialize();
    void tick();

private:
    // core member
    std::unique_ptr<interface::Window> window_;
    std::unique_ptr<interface::camera> camera_;
    window_config window_config_;
};