#ifndef MATERIAL_HPP
#define MATERIAL_HPP
#include "Vec.hpp"

struct Material
{
	float refractive_index;
	Vec4f albedo;
	Vec3f diffuse;
	float specular_exp;
};

#endif //MATERIAL_HPP
