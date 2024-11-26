#pragma once

#include <cmath>
#include <limits>

namespace raym
{

  struct vec3;
  struct vec4;
  struct mat4;

  constexpr float sqrt_helper(float x, float curr, float prev)
  {
    return curr == prev ? curr : sqrt_helper(x, 0.5f * (curr + x / curr), curr);
  }

  constexpr float sqrt(float x)
  {
    return x >= 0 && x < std::numeric_limits<float>::infinity()
               ? sqrt_helper(x, x, 0)
               : std::numeric_limits<float>::quiet_NaN();
  }

  struct vec3
  {
    float x, y, z;

    constexpr vec3() : x(0), y(0), z(0) {}
    constexpr vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    constexpr vec3(float v) : x(v), y(v), z(v) {}

    constexpr vec3 operator+(const vec3 &other) const
    {
      return vec3(x + other.x, y + other.y, z + other.z);
    }

    constexpr vec3 operator-(const vec3 &other) const
    {
      return vec3(x - other.x, y - other.y, z - other.z);
    }

    constexpr vec3 operator*(const vec3 &other) const
    {
      return vec3(
          x * other.x,
          y * other.y,
          z * other.z);
    }

    constexpr vec3 operator*(float scalar) const
    {
      return vec3(x * scalar, y * scalar, z * scalar);
    }

    friend constexpr vec3 operator*(float scalar, const vec3 &v)
    {
      return vec3(scalar * v.x, scalar * v.y, scalar * v.z);
    }

    constexpr vec3 operator/(float scalar) const
    {
      return vec3(x / scalar, y / scalar, z / scalar);
    }

    constexpr bool operator==(const vec3 &other) const
    {
      return x == other.x && y == other.y && z == other.z;
    }

    constexpr float &operator[](int index)
    {
      return index == 0 ? x : (index == 1 ? y : z);
    }

    constexpr const float &operator[](int index) const
    {
      return index == 0 ? x : (index == 1 ? y : z);
    }
  };

  constexpr float dot(const vec3 &a, const vec3 &b)
  {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  constexpr vec3 cross(const vec3 &a, const vec3 &b)
  {
    return vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
  }

  constexpr float length(const vec3 &v)
  {
    return sqrt(dot(v, v));
  }

  constexpr vec3 normalize(const vec3 &v)
  {
    float len = length(v);
    return vec3(v.x / len, v.y / len, v.z / len);
  }

  constexpr vec3 reflect(const vec3 &I, const vec3 &N)
  {
    return I - N * (2.0f * dot(N, I));
  }

  constexpr float clamp(float x, float minVal, float maxVal)
  {
    return x < minVal ? minVal : (x > maxVal ? maxVal : x);
  }

  constexpr vec3 clamp(const vec3 &v, const vec3 &minVal, const vec3 &maxVal)
  {
    return vec3(
        clamp(v.x, minVal.x, maxVal.x),
        clamp(v.y, minVal.y, maxVal.y),
        clamp(v.z, minVal.z, maxVal.z));
  }

  constexpr float max(float a, float b)
  {
    return a > b ? a : b;
  }

  constexpr float distance(const vec3 &a, const vec3 &b)
  {
    return length(a - b);
  }

  constexpr float pow(float base, int exponent)
  {
    return exponent == 0 ? 1 : exponent > 0 ? base * pow(base, exponent - 1)
                                            : 1 / pow(base, -exponent);
  }

  constexpr float pow(float base, float exponent)
  {
    return std::pow(base, exponent); // Requires C++20 for constexpr std::pow
  }

  constexpr vec3 pow(const vec3 &base, float exponent)
  {
    return vec3(
        pow(base.x, exponent),
        pow(base.y, exponent),
        pow(base.z, exponent));
  }

  constexpr vec3 pow(const vec3 &base, const vec3 &exponent)
  {
    return vec3(
        pow(base.x, exponent.x),
        pow(base.y, exponent.y),
        pow(base.z, exponent.z));
  }

  struct vec4
  {
    float x, y, z, w;

    // Constructors
    constexpr vec4() : x(0), y(0), z(0), w(0) {}
    constexpr vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    constexpr vec4(const vec3 &v, float w) : x(v.x), y(v.y), z(v.z), w(w) {}

    // Operator overloads
    constexpr vec4 operator+(const vec4 &other) const
    {
      return vec4(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    constexpr vec4 operator-(const vec4 &other) const
    {
      return vec4(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    constexpr vec4 operator*(float scalar) const
    {
      return vec4(x * scalar, y * scalar, z * scalar, w * scalar);
    }

    // Scalar multiplication (scalar * vec4)
    friend constexpr vec4 operator*(float scalar, const vec4 &v)
    {
      return vec4(scalar * v.x, scalar * v.y, scalar * v.z, scalar * v.w);
    }

    constexpr vec4 operator/(float scalar) const
    {
      return vec4(x / scalar, y / scalar, z / scalar, w / scalar);
    }

    constexpr bool operator==(const vec4 &other) const
    {
      return x == other.x && y == other.y && z == other.z && w == other.w;
    }

    // Access operators
    constexpr float &operator[](int index)
    {
      return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w));
    }

    constexpr const float &operator[](int index) const
    {
      return index == 0 ? x : (index == 1 ? y : (index == 2 ? z : w));
    }
  };

  // Dot product for vec4
  constexpr float dot(const vec4 &a, const vec4 &b)
  {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  }

  // Length of a vec4
  constexpr float length(const vec4 &v)
  {
    return sqrt(dot(v, v));
  }

  // Normalize a vec4
  constexpr vec4 normalize(const vec4 &v)
  {
    float len = length(v);
    return vec4(v.x / len, v.y / len, v.z / len, v.w / len);
  }

  // Clamp function for vec4
  constexpr vec4 clamp(const vec4 &v, const vec4 &minVal, const vec4 &maxVal)
  {
    return vec4(
        clamp(v.x, minVal.x, maxVal.x),
        clamp(v.y, minVal.y, maxVal.y),
        clamp(v.z, minVal.z, maxVal.z),
        clamp(v.w, minVal.w, maxVal.w));
  }

  // Max function for vec4
  constexpr vec4 max(const vec4 &a, const vec4 &b)
  {
    return vec4(
        max(a.x, b.x),
        max(a.y, b.y),
        max(a.z, b.z),
        max(a.w, b.w));
  }

  struct mat4
  {
    float data[4][4];

    constexpr mat4() : data{
                           {1, 0, 0, 0}, // Column 0
                           {0, 1, 0, 0}, // Column 1
                           {0, 0, 1, 0}, // Column 2
                           {0, 0, 0, 1}  // Column 3
                       }
    {
    }

    constexpr mat4(float diagonal) : data{
                                         {diagonal, 0, 0, 0},
                                         {0, diagonal, 0, 0},
                                         {0, 0, diagonal, 0},
                                         {0, 0, 0, diagonal}} {}

    constexpr float *operator[](int index)
    {
      return data[index];
    }

    constexpr const float *operator[](int index) const
    {
      return data[index];
    }

    constexpr mat4 operator*(const mat4 &other) const
    {
      mat4 result;
      for (int col = 0; col < 4; ++col)
      {
        for (int row = 0; row < 4; ++row)
        {
          result.data[col][row] = 0;
          for (int k = 0; k < 4; ++k)
          {
            result.data[col][row] += data[k][row] * other.data[col][k];
          }
        }
      }
      return result;
    }

    constexpr vec3 operator*(const vec3 &v) const
    {
      float x = data[0][0] * v.x + data[1][0] * v.y + data[2][0] * v.z + data[3][0];
      float y = data[0][1] * v.x + data[1][1] * v.y + data[2][1] * v.z + data[3][1];
      float z = data[0][2] * v.x + data[1][2] * v.y + data[2][2] * v.z + data[3][2];
      float w = data[0][3] * v.x + data[1][3] * v.y + data[2][3] * v.z + data[3][3];
      return vec3(x / w, y / w, z / w);
    }

    constexpr vec4 operator*(const vec4 &v) const
    {
      float x = data[0][0] * v.x + data[1][0] * v.y + data[2][0] * v.z + data[3][0] * v.w;
      float y = data[0][1] * v.x + data[1][1] * v.y + data[2][1] * v.z + data[3][1] * v.w;
      float z = data[0][2] * v.x + data[1][2] * v.y + data[2][2] * v.z + data[3][2] * v.w;
      float w = data[0][3] * v.x + data[1][3] * v.y + data[2][3] * v.z + data[3][3] * v.w;
      return vec4(x, y, z, w);
    }
  };

  constexpr mat4 transpose(const mat4 &m)
  {
    mat4 result;
    for (int col = 0; col < 4; ++col)
    {
      for (int row = 0; row < 4; ++row)
      {
        result.data[row][col] = m.data[col][row];
      }
    }
    return result;
  }

  constexpr raym::mat4 inverse(const raym::mat4 &m)
  {
    const float *a = &m[0][0];

    float inv[16] = {};

    inv[0] = a[5] * a[10] * a[15] -
             a[5] * a[11] * a[14] -
             a[9] * a[6] * a[15] +
             a[9] * a[7] * a[14] +
             a[13] * a[6] * a[11] -
             a[13] * a[7] * a[10];

    inv[4] = -a[4] * a[10] * a[15] +
             a[4] * a[11] * a[14] +
             a[8] * a[6] * a[15] -
             a[8] * a[7] * a[14] -
             a[12] * a[6] * a[11] +
             a[12] * a[7] * a[10];

    inv[8] = a[4] * a[9] * a[15] -
             a[4] * a[11] * a[13] -
             a[8] * a[5] * a[15] +
             a[8] * a[7] * a[13] +
             a[12] * a[5] * a[11] -
             a[12] * a[7] * a[9];

    inv[12] = -a[4] * a[9] * a[14] +
              a[4] * a[10] * a[13] +
              a[8] * a[5] * a[14] -
              a[8] * a[6] * a[13] -
              a[12] * a[5] * a[10] +
              a[12] * a[6] * a[9];

    inv[1] = -a[1] * a[10] * a[15] +
             a[1] * a[11] * a[14] +
             a[9] * a[2] * a[15] -
             a[9] * a[3] * a[14] -
             a[13] * a[2] * a[11] +
             a[13] * a[3] * a[10];

    inv[5] = a[0] * a[10] * a[15] -
             a[0] * a[11] * a[14] -
             a[8] * a[2] * a[15] +
             a[8] * a[3] * a[14] +
             a[12] * a[2] * a[11] -
             a[12] * a[3] * a[10];

    inv[9] = -a[0] * a[9] * a[15] +
             a[0] * a[11] * a[13] +
             a[8] * a[1] * a[15] -
             a[8] * a[3] * a[13] -
             a[12] * a[1] * a[11] +
             a[12] * a[3] * a[9];

    inv[13] = a[0] * a[9] * a[14] -
              a[0] * a[10] * a[13] -
              a[8] * a[1] * a[14] +
              a[8] * a[2] * a[13] +
              a[12] * a[1] * a[10] -
              a[12] * a[2] * a[9];

    inv[2] = a[1] * a[6] * a[15] -
             a[1] * a[7] * a[14] -
             a[5] * a[2] * a[15] +
             a[5] * a[3] * a[14] +
             a[13] * a[2] * a[7] -
             a[13] * a[3] * a[6];

    inv[6] = -a[0] * a[6] * a[15] +
             a[0] * a[7] * a[14] +
             a[4] * a[2] * a[15] -
             a[4] * a[3] * a[14] -
             a[12] * a[2] * a[7] +
             a[12] * a[3] * a[6];

    inv[10] = a[0] * a[5] * a[15] -
              a[0] * a[7] * a[13] -
              a[4] * a[1] * a[15] +
              a[4] * a[3] * a[13] +
              a[12] * a[1] * a[7] -
              a[12] * a[3] * a[5];

    inv[14] = -a[0] * a[5] * a[14] +
              a[0] * a[6] * a[13] +
              a[4] * a[1] * a[14] -
              a[4] * a[2] * a[13] -
              a[12] * a[1] * a[6] +
              a[12] * a[2] * a[5];

    inv[3] = -a[1] * a[6] * a[11] +
             a[1] * a[7] * a[10] +
             a[5] * a[2] * a[11] -
             a[5] * a[3] * a[10] -
             a[9] * a[2] * a[7] +
             a[9] * a[3] * a[6];

    inv[7] = a[0] * a[6] * a[11] -
             a[0] * a[7] * a[10] -
             a[4] * a[2] * a[11] +
             a[4] * a[3] * a[10] +
             a[8] * a[2] * a[7] -
             a[8] * a[3] * a[6];

    inv[11] = -a[0] * a[5] * a[11] +
              a[0] * a[7] * a[9] +
              a[4] * a[1] * a[11] -
              a[4] * a[3] * a[9] -
              a[8] * a[1] * a[7] +
              a[8] * a[3] * a[5];

    inv[15] = a[0] * a[5] * a[10] -
              a[0] * a[6] * a[9] -
              a[4] * a[1] * a[10] +
              a[4] * a[2] * a[9] +
              a[8] * a[1] * a[6] -
              a[8] * a[2] * a[5];

    float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];

    if (det == 0)
      return raym::mat4(0.0f); // Non-invertible matrix; return zero matrix or handle as needed

    float invDet = 1.0f / det;

    raym::mat4 invOut;

    for (int i = 0; i < 16; i++)
      invOut[i / 4][i % 4] = inv[i] * invDet;

    return invOut;
  }

  constexpr vec4 to_vec4(const vec3 &v, float w)
  {
    return vec4(v.x, v.y, v.z, w);
  }

  constexpr vec3 to_vec3(const vec4 &v)
  {
    return vec3(v.x, v.y, v.z);
  }

  constexpr vec3 min(const vec3 &v, const vec3 &u)
  {
    return vec3(
        v.x < u.x ? v.x : u.x,
        v.y < u.y ? v.y : u.y,
        v.z < u.z ? v.z : u.z);
  }

  constexpr vec3 max(const vec3 &v, const vec3 &u)
  {
    return vec3(
        v.x > u.x ? v.x : u.x,
        v.y > u.y ? v.y : u.y,
        v.z > u.z ? v.z : u.z);
  }

  constexpr vec3 to_vec3_perspective(const vec4 &v)
  {
    return vec3(v.x / v.w, v.y / v.w, v.z / v.w);
  }

  constexpr vec3 to_vec3_perspective_safe(const vec4 &v)
  {
    return (v.w != 0.0f)
               ? vec3(v.x / v.w, v.y / v.w, v.z / v.w)
               : vec3(v.x, v.y, v.z);
  }

  constexpr mat4 translate(const vec3 &translation)
  {
    mat4 result(1.0f);
    result[3][0] = translation.x;
    result[3][1] = translation.y;
    result[3][2] = translation.z;
    return result;
  }

  constexpr mat4 scale(const vec3 &scalingFactors)
  {
    mat4 result(1.0f);
    result[0][0] = scalingFactors.x;
    result[1][1] = scalingFactors.y;
    result[2][2] = scalingFactors.z;
    return result;
  }

} // namespace raym
