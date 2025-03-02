#ifndef LIGHT_HPP
#define LIGHT_HPP
#include "Vec.hpp"

struct Light
{
	Vec3f position{};
	float intensity{0};
};

#endif //LIGHT_HPP
