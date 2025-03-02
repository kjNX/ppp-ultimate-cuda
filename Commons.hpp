#ifndef SETTINGS_HPP
#define SETTINGS_HPP
#include <cmath>
#include <vector>
#include <fstream>
#include <thread>

#include "Vec.hpp"

namespace Commons
{
	inline constexpr const char* const& OUTPUT_FILENAME{"out.ppm"};

	// display settings
	inline constexpr size_t RENDER_WIDTH{1024u};
	inline constexpr size_t RENDER_HEIGHT{768u};
	inline constexpr size_t RENDER_PIXELS{RENDER_WIDTH * RENDER_HEIGHT};
	inline constexpr float ASPECT_RATIO{RENDER_WIDTH / static_cast<float>(RENDER_HEIGHT)};
	inline constexpr float FOV{M_PI / 2.};

	// detail settings
	inline constexpr float DRAW_DISTANCE{1000};
	inline constexpr size_t MAX_REFLECTIONS{5};

	__device__ inline constexpr Vec3f VEC3F_ZERO{0, 0, 0};

	__device__ inline constexpr Vec3f BG_COLOR{VEC3F_ZERO};
	__device__ inline constexpr Vec3f RENDER_ORIGIN{VEC3F_ZERO};

	__host__ void ppmWriteBuffer(const char* const& filename, const Vec3f* const& fb);
}

#endif //SETTINGS_HPP
