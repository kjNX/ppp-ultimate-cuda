#ifndef SPHERE_HPP
#define SPHERE_HPP
#include "Vec.hpp"

//#include "Material.hpp"

struct Sphere
{
	Vec3f center;
	float radius;
	Material material;

	__device__ bool rayIntersects(const Vec3f& origin, const Vec3f& direction, float& t0) const
	{
		const Vec3f& L{center - origin};
		const float& tca{L * direction};
		const float& d2{L * L - tca * tca};
		if(d2 > radius * radius) return false;
		const float& thc{sqrtf(radius * radius - d2 * d2)};
		t0 = tca - thc;
		const float& t1{tca + thc};
		if(t0 < 0) t0 = t1;
		if(t0 < 0) return false;
		return true;
	}
};

#endif //SPHERE_HPP
