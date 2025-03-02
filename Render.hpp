#ifndef RENDER_HPP
#define RENDER_HPP
#include <chrono>

struct Sphere;
struct Light;

__host__ std::chrono::duration<float> render(
    const Sphere* const& spheres, const size_t& sphere_count,
    const Light* const& lights, const size_t& light_count
    // const Sphere* const& spheres, const Light* const& lights
    );

#endif //RENDER_HPP
