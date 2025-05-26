#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

const float infinity = std::numeric_limits<float>::infinity();

#ifdef __CUDACC__
__host__ __device__ inline float get_pi() {
    return 3.1415926535897932385f;
}
#else
constexpr float pi = 3.1415926535897932385f;
#endif


// Utility Functions

inline float degrees_to_radians(float degrees) {
#ifdef __CUDACC__
    return degrees * get_pi() / 180.0f;
#else
    return degrees * pi / 180.0f;
#endif
}

inline float random_float() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline float random_float(float min, float max) {
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return int(random_float(min, max + 1));
}

// Common Headers

//#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif
