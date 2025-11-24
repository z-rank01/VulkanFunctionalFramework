#include <memory>

#include "_interface/camera.h"
#include "_interface/window.h"

struct SWindowConfig
{
    int width_;
    int height_;
    std::string title_;

    [[nodiscard]] constexpr auto validate() const -> bool { return width_ > 0 && height_ > 0; }
};


class AppSample
{
public:
    // core public function
    void initialize();
    void tick();

private:
    // core member
    std::unique_ptr<interface::Window> window_;
    std::unique_ptr<interface::Camera> camera_;
    SWindowConfig window_config_;
};