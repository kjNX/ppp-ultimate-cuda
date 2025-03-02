#include <chrono>

#include "Vec.hpp"
#include "Light.hpp"
#include "Material.hpp"
#include "Sphere.hpp"
#include "Render.hpp"

int main()
{
	// int background_width, background_height;
	// unsigned char* const background{stbi_load("bg.jpg",
		// &background_width, &background_height, nullptr, 3)};

	const size_t& material_count{4};
	const Material* const& materials{new Material[material_count] {
		{ 1,
			{
				.6,
				.3,
				.1,
				0
			}, {
				.4,
				.4,
				.3
			}, 50
		},{ 1.5,
			{
				0,
				.5,
				.1,
				.8
			}, {
				.6,
				.7,
				.8
			}, 125
		}, { 1,
			{
				.9,
				.1,
				0,
				0
			}, {
				.3,
				.1,
				.1
			}, 10
		}, { 1,
			{
				0,
				10,
				.8,
				0
			}, {
				1,
				1,
				1
			}, 1425
		}
	}};

	// Sphere sp1{{0, 0, -16}, 5};

	const size_t& sphere_count{4};
	const Sphere* const& spheres{new Sphere[sphere_count] {
		{
			{
				-3,
				0,
				-8,
			}, 2, materials[0]
		}, {
				{
					-1,
					-1.5,
					-6,
				}, 2, materials[1]
			}, {
				{
					1.5,
					-.5,
					-9
				}, 3, materials[2]
			}, {
				{
					7,
					5,
					-9
				}, 4, materials[3]
			}
	}};

	const size_t& light_count{3};
	const Light* const lights{new Light[light_count] {
		{
			{
				-20,
				20,
				10
			}, 1.5
		}, {
				{
					30,
					50,
					-12.5
				}, 1.8
			}, {
				{
					30,
					20,
					15
				}, 1.7
			}
	}};

	const std::chrono::duration<float>& render_time{render(spheres, sphere_count, lights, light_count)};
	printf("Done! Elapsed time: %f seconds.\n", render_time.count());

	delete[] materials;
	delete[] spheres;
	delete[] lights;

	return 0;
}