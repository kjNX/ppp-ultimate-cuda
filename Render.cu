#include <chrono>

#include "Commons.hpp"
#include "Light.hpp"
#include "Material.hpp"
#include "Sphere.hpp"

using clk = std::chrono::steady_clock;
__device__ Vec3f reflect(const Vec3f& i, const Vec3f& N) { return i - N * 2.f * (i * N); }

__device__ Vec3f refract(const Vec3f& i, const Vec3f& N, const float& eta_t, const float& eta_i = 1.f)
{
	const float& cos_i{-max(-1.f, min(1.f, i * N))};
	if(cos_i < 0) refract(i, -N, eta_i, eta_t);
	const float& eta{eta_i / eta_t};
	const float& k{1 - eta * eta * (1 - cos_i * cos_i)};
	return k < 0 ? Vec3f{0, 0, 0} : i * eta + N * (eta * cos_i - sqrtf(k));
}

__device__ size_t scene_intersect(const Vec3f& origin, const Vec3f& direction,
	const Sphere* const& spheres, const size_t& sphere_count,
	Vec3f& hit, Vec3f& N
	)
{
	float min_dist{MAXFLOAT};
	size_t temp{SIZE_MAX};
	for(size_t i{0u}; i < sphere_count; ++i)
	{
		if(float dist_i{}; spheres[i].rayIntersects(origin, direction, dist_i) && dist_i < min_dist)
		{
			min_dist = dist_i;
			temp = i;
		}
	}
	if(temp >= sphere_count) return SIZE_MAX;
	hit = origin + direction * min_dist;
	N = (hit - spheres[temp].center).normalize();
	return min_dist < Commons::DRAW_DISTANCE ? temp : SIZE_MAX;
}

__device__ Vec3f cast_ray(const Vec3f& origin, const Vec3f& direction,
	const Sphere* const& spheres, const size_t& sphere_count,
	const Light* const& lights, const size_t& light_count,
	// const unsigned char* const& bg, const int& bg_width, const int& bg_height,
	const size_t& depth = 0
	)
{
	Vec3f hit{}, N{};
	const auto& origin_func{[&hit, &N](const Vec3f& dir)
		{ return hit - N * (dir * N < 0 ? 1e-3f : -1e-3f); }};

	const size_t& idx{scene_intersect(origin, direction, spheres, sphere_count, hit, N)};
	// printf("%lu\n", idx);
	if(depth > Commons::MAX_REFLECTIONS || idx == SIZE_MAX || idx >= sphere_count) return Commons::BG_COLOR;
	// if(sceneIntersect(origin, direction, spheres)) return Commons::BG_COLOR;
	/*
	{
		int base_idx{static_cast<int>(bg_width * (direction.x / 2.f + .5) + bg_height * (direction.y / 2.f + .5))};
		return Vec3f{bg[base_idx] / 255.f, bg[base_idx + 1] / 255.f, bg[base_idx + 2] / 255.f};
	}
	*/

	const Vec3f& reflect_direction{reflect(direction, N).normalize()};
	const Vec3f& reflect_origin{origin_func(reflect_direction)};
	const Vec3f& reflect_color{cast_ray(reflect_origin, reflect_direction, spheres, sphere_count, lights, light_count
		// , bg, bg_width, bg_height
		, depth + 1)};

	// const Vec3f& refract_direction{refract(direction, N, spheres[idx].material.refractive_index).normalize()};
	// const Vec3f& refract_origin{origin_func(refract_direction)};
	// const Vec3f& refract_color{cast_ray(hit, refract_direction, spheres, sphere_count, lights, light_count
		// , bg, bg_width, bg_height
		// , depth + 1)};

	const Material& mat{spheres[idx].material};

	float diffuse_light{0.f};
	float specular_light{0.f};
	for(size_t i{0u}; i < light_count; ++i)
	{
		const auto& [position, intensity] = lights[i];
		Vec3f light_direction{(position - hit).normalize()};
		const float& light_distance{(position - hit).norm()};

		// const Vec3f& shadow_origin{origin_func(light_direction)};
		Vec3f shadow_hit{}, shadow_N{};
		if(const size_t& shadow_idx{scene_intersect(hit, light_direction, spheres, sphere_count, shadow_hit, shadow_N)};
			shadow_idx < sphere_count && idx != shadow_idx && (shadow_hit - hit).norm() < light_distance)
			continue;

		diffuse_light += intensity * max(0.f, light_direction * N);
		specular_light += powf(max(0.f, reflect(light_direction, N) * direction), mat.specular_exp)
		* intensity;
	}

	return mat.diffuse * diffuse_light * mat.albedo.x +
		Vec3f{1, 1, 1} * specular_light * mat.albedo.y +
			reflect_color * mat.albedo.z;// +
				// refract_color * mat.albedo.w;
}

__global__ void render(Vec3f* const& fb,
	const Sphere* const& spheres, const size_t& sphere_count,
	const Light* const& lights, const size_t& light_count)
{
	const size_t& i{threadIdx.x + blockIdx.x * blockDim.x};
	const float& x{static_cast<float>(i % Commons::RENDER_WIDTH) + .5f - Commons::RENDER_WIDTH / 2.f};
	const float& y{-static_cast<float>(i) / Commons::RENDER_WIDTH - .5f + Commons::RENDER_HEIGHT / 2.f};
	const float& z{-static_cast<float>(Commons::RENDER_HEIGHT) / (2 * tanf(Commons::FOV / 2.f))};
	fb[i] = cast_ray(Commons::RENDER_ORIGIN, Vec3f{x, y, z}.normalize(),
		spheres, sphere_count, lights, light_count
		// , lights, bg, bg_width, bg_height
		);

	/*
	for(size_t i{0u}; i < Commons::RENDER_PIXELS; ++i)
	{
	const float& x{static_cast<float>(i % Commons::RENDER_WIDTH) + .5f - Commons::RENDER_WIDTH / 2.f};
	const float& y{-static_cast<float>(i) / Commons::RENDER_WIDTH - .5f + Commons::RENDER_HEIGHT / 2.f};
	const float& z{-static_cast<float>(Commons::RENDER_HEIGHT) / (2 * tanf(Commons::FOV / 2.f))};
	fb[i] = cast_ray(Commons::RENDER_ORIGIN, Vec3f{x, y, z}.normalize(),
		spheres, sphere_count, lights, light_count
		// , lights, bg, bg_width, bg_height
		);
	}
	*/

	// fb[i] = {static_cast<float>(i) / Commons::RENDER_WIDTH / Commons::RENDER_HEIGHT,
		// static_cast<float>(i % Commons::RENDER_WIDTH) / static_cast<float>(Commons::RENDER_WIDTH), 0};
}

__host__ std::chrono::duration<float> render(
	const Sphere* const& spheres, const size_t& sphere_count,
	const Light* const& lights, const size_t& light_count
	// const unsigned char* const& bg, const int& bg_width, const int& bg_height,
	)
{
	cudaError_t err{cudaSuccess};
	Vec3f* d_fb{};
	err = cudaMalloc((void**) &d_fb, Commons::RENDER_PIXELS * sizeof(Vec3f));

	if(err != cudaSuccess)
	{
		printf("Failed to allocate framebuffer memory.\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	const size_t& blockSize{512u};
	const size_t& blockCount{(Commons::RENDER_PIXELS + blockSize - 1) / blockSize};
	const auto& startTime{clk::now()};
	render<<<blockCount, blockSize>>>(d_fb, spheres, sphere_count, lights, light_count);
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	if(err != cudaSuccess)
	{
		printf("Failed to render to framebuffer.\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	const auto& endTime{clk::now()};
	Vec3f* h_fb{(Vec3f*)malloc(Commons::RENDER_PIXELS * sizeof(Vec3f))};
	err = cudaMemcpy(h_fb, d_fb, Commons::RENDER_PIXELS * sizeof(Vec3f), cudaMemcpyDeviceToHost);

	if(err != cudaSuccess)
	{
		printf("Failed to copy framebuffer to CPU.\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	Commons::ppmWriteBuffer(Commons::OUTPUT_FILENAME, h_fb);
	free(h_fb);
	err = cudaFree(d_fb);

	if(err != cudaSuccess)
	{
		printf("Failed to free framebuffer memory.\n");
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}

	return endTime - startTime;
}
