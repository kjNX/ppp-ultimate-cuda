#ifndef VECTOR_HPP
#define VECTOR_HPP

template <size_t COMP, typename T> struct Vec
{
	T data[COMP]{};

	__host__ __device__ T& operator[](const size_t& i) { return data[i]; }
	__host__ __device__ const T& operator[](const size_t& i) const { return data[i]; }
};

typedef Vec<2, float> Vec2f;
typedef Vec<3, float> Vec3f;
typedef Vec<3, int> Vec3i;
typedef Vec<4, float> Vec4f;

template<typename T> struct Vec<2, T>
{
	T x, y;

	__host__ __device__ constexpr Vec() : x{}, y{} {}
	__host__ __device__ constexpr Vec(const T& a, const T& b) : x{a}, y{b} {}

	__host__ __device__ T& operator[](const size_t& i) { return i <= 0 ? x : y; }
	__host__ __device__ const T& operator[](const size_t& i) const { return i <= 0 ? x : y; }
};

template<typename T> struct Vec<3, T>
{
	T x, y, z;

	__host__ __device__ constexpr Vec() : x{}, y{}, z{} {}
	__host__ __device__ constexpr Vec(const T& a, const T& b, const T& c) : x{a}, y{b}, z{c} {}

 	__device__ float norm() { return sqrtf(x * x + y * y + z * z); }
	__device__ Vec& normalize(T l = 1)
	{
		float temp{norm()};
		*this = *this * (l / (temp ? temp : 1));
		return *this;
	}

	__host__ __device__ T& operator[](const size_t& i) { return i <= 0 ? x : i == 1 ? y : z; }
	__host__ __device__ const T& operator[](const size_t& i) const { return i <= 0 ? x : i == 1 ? y : z; }
};

template<typename T> struct Vec<4, T>
{
	T x, y, z, w;

	__host__ __device__ constexpr Vec() : x{}, y{}, z{}, w{} {}
	__host__ __device__ constexpr Vec(const T& a, const T& b, const T& c, const T& d) : x{a}, y{b}, z{c}, w{d} {}

	__host__ __device__ T& operator[](const size_t& i) { return i <= 0 ? x : i == 1 ? y : i == 2 ? z : w; }
	__host__ __device__ const T& operator[](const size_t& i) const { return i <= 0 ? x : i == 1 ? y : i == 2 ? z : w; }
};

template<size_t COMP, typename T> __device__ T operator*(const Vec<COMP, T>& a, const Vec<COMP, T>& b)
{
	T temp{};
	for(size_t i{0u}; i < COMP; ++i) temp += a[i] * b[i];
	return temp;
}

template<size_t COMP, typename T> __device__ Vec<COMP, T> operator*(Vec<COMP, T> a, const T& b)
{
	for(size_t i{0u}; i < COMP; ++i) a[i] *= b;
	return a;
}

template<size_t COMP, typename T> __device__ Vec<COMP, T> operator-(const Vec<COMP, T>& a) { return a * T(-1); }

template<size_t COMP, typename T> __device__ Vec<COMP, T> operator+(Vec<COMP, T> a, const Vec<COMP, T>& b)
{
	for(size_t i{0u}; i < COMP; ++i) a[i] += b[i];
	return a;
}

template<size_t COMP, typename T> __device__ Vec<COMP, T> operator-(const Vec<COMP, T>& a, const Vec<COMP, T>& b)
{ return a + -b; }

template<typename T> Vec<3, T> __device__ cross(const Vec<3, T>& v1, const Vec<3, T>& v2)
{ return {v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x}; }

#endif //VECTOR_HPP
