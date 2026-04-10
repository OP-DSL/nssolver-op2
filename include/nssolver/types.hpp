#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace nssolver {

using Real = double;
using Index = std::int32_t;

struct Vec3 {
    Real x {};
    Real y {};
    Real z {};
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(Real s, const Vec3& v) {
    return {s * v.x, s * v.y, s * v.z};
}

inline Vec3 operator/(const Vec3& v, Real s) {
    return {v.x / s, v.y / s, v.z / s};
}

inline Real dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Real norm(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

}  // namespace nssolver
