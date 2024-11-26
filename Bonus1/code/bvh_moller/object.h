#pragma once
#include <array>

#include "material.h"
#include "../ray_math.h"
#include <cfloat>

struct Ray
{
public:
  raym::vec3 origin;
  raym::vec3 direction;

  Ray(raym::vec3 origin, raym::vec3 direction) : origin(origin), direction(direction) {}
};

class AABB
{
public:
  raym::vec3 minBounds;
  raym::vec3 maxBounds;

  AABB() : minBounds(raym::vec3(FLT_MAX)), maxBounds(raym::vec3(-FLT_MAX)) {}
  AABB(const raym::vec3 &min, const raym::vec3 &max) : minBounds(min), maxBounds(max) {}

  void expand(const raym::vec3 &point)
  {
    minBounds = raym::min(minBounds, point);
    maxBounds = raym::max(maxBounds, point);
  }

  void merge(const AABB &other)
  {
    minBounds = raym::min(minBounds, other.minBounds);
    maxBounds = raym::max(maxBounds, other.maxBounds);
  }

  bool intersect(const Ray &ray)
  {
    float txmin = (minBounds.x - ray.origin.x) / ray.direction.x;
    float txmax = (maxBounds.x - ray.origin.x) / ray.direction.x;
    if (txmin > txmax)
      std::swap(txmin, txmax);

    float tymin = (minBounds.y - ray.origin.y) / ray.direction.y;
    float tymax = (maxBounds.y - ray.origin.y) / ray.direction.y;
    if (tymin > tymax)
      std::swap(tymin, tymax);

    float tzmin = (minBounds.z - ray.origin.z) / ray.direction.z;
    float tzmax = (maxBounds.z - ray.origin.z) / ray.direction.z;
    if (tzmin > tzmax)
      std::swap(tzmin, tzmax);

    int overlapCount = 0;

    if (txmin <= tymax && txmax >= tymin)
    {
      overlapCount++;
    }

    if (txmin <= tzmax && txmax >= tzmin)
    {
      overlapCount++;
    }

    if (tymin <= tzmax && tymax >= tzmin)
    {
      overlapCount++;
    }

    return overlapCount >= 2;
  }
};

class Object;

struct Hit
{
  bool hit;
  raym::vec3 normal;
  raym::vec3 intersection;
  float distance;
  Object *object;
};

class Object
{
protected:
  raym::mat4 transformationMatrix;
  raym::mat4 inverseTransformationMatrix;
  raym::mat4 normalMatrix;

public:
  raym::vec3 color;
  Material material;

  virtual Hit intersect(const Ray &ray) = 0;

  constexpr void setTransformation(const raym::mat4 &matrix)
  {
    transformationMatrix = matrix;
    inverseTransformationMatrix = raym::inverse(matrix);
    normalMatrix = raym::transpose(inverseTransformationMatrix);
  }
};

class Plane : public Object
{

private:
  raym::vec3 normal;
  raym::vec3 point;

public:
  Plane(const raym::vec3 &point, const raym::vec3 &normal) : point(point), normal(normal) {}
  Plane(const raym::vec3 &point, const raym::vec3 &normal, const Material &material) : point(point), normal(normal)
  {
    this->material = material;
  }

  // ! constexpr fa si che questi calcoli li faccia il compilatore e non il programma
  constexpr float _DdotN(const Ray &ray)
  {
    return raym::dot(ray.direction, normal);
  }

  constexpr float _PdotN(const Ray &ray)
  {
    return raym::dot(point - ray.origin, normal);
  }

  // ! Uso le constexpr per PdotN e NdotN
  Hit intersect(const Ray &ray) override
  {
    Hit hit;
    hit.hit = false;

    float DdotN = _DdotN(ray);
    if (DdotN < 0)
    {

      float PdotN = _PdotN(ray);
      float t = PdotN / DdotN;

      if (t > 0)
      {
        hit.hit = true;
        hit.normal = normal;
        hit.distance = t;
        hit.object = this;
        hit.intersection = t * ray.direction + ray.origin;
      }
    }

    return hit;
  }
};

class Triangle : public Object
{
private:
  std::array<raym::vec3, 3> vertices;
  std::array<raym::vec3, 3> normals;
  bool smoothSmoothing;

public:
  AABB boundingBox;

  constexpr raym::vec3 _triangle_normal()
  {
    return raym::normalize(raym::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
  }

  Triangle(const std::array<raym::vec3, 3> &vertices, const Material &material) : vertices(vertices)
  {
    this->material = material;
    smoothSmoothing = false;
    calculateBoundingBox();
  }

  Triangle(const std::array<raym::vec3, 3> &vertices, const std::array<raym::vec3, 3> &normals, const Material &material) : vertices(vertices), normals(normals)
  {
    this->material = material;
    smoothSmoothing = true;
    calculateBoundingBox();
  }

  constexpr raym::vec3 _tOrigin(const Ray &ray)
  {
    return raym::to_vec3(inverseTransformationMatrix * raym::vec4(ray.origin, 1.0));
  }

  constexpr raym::vec3 _tDirection(const Ray &ray)
  {
    return raym::to_vec3(raym::normalize(inverseTransformationMatrix * raym::vec4(ray.direction, 0.0)));
  }

  constexpr raym::vec3 _Ncross()
  {
    return raym::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
  }

  constexpr raym::vec3 _ns(const int &i, const raym::vec3 &p)
  {
    return raym::cross(vertices[i] - p, vertices[(i + 1) % 3] - p);
  }

  constexpr float _sign(const raym::vec3 &N, const raym::vec3 &n)
  {
    return raym::dot(N, n);
  }

  constexpr raym::vec3 _hitIntersection(const Hit &hit)
  {
    return raym::to_vec3(transformationMatrix * raym::vec4(hit.intersection, 1.0));
  }

  constexpr raym::vec3 _hitNormal(const Hit &hit)
  {
    return raym::to_vec3(raym::normalize((normalMatrix * raym::vec4(hit.normal, 0.0f))));
  }

  constexpr float _hitDistance(const Hit &hit, const Ray &ray)
  {
    return raym::length(hit.intersection - ray.origin);
  }

  Hit intersect(const Ray &ray) override
  {
    Hit hit;
    hit.hit = false;

    raym::vec3 tOrigin = _tOrigin(ray);
    raym::vec3 tDirection = _tDirection(ray);

    const raym::vec3 &v0 = vertices[0];
    const raym::vec3 &v1 = vertices[1];
    const raym::vec3 &v2 = vertices[2];

    const float EPSILON = 1e-8f;
    raym::vec3 edge1 = v1 - v0;
    raym::vec3 edge2 = v2 - v0;

    raym::vec3 h = raym::cross(tDirection, edge2);
    float a = raym::dot(edge1, h);

    if (std::abs(a) < EPSILON)
      return hit;

    float f = 1.0f / a;
    raym::vec3 s = tOrigin - v0;
    float u = f * raym::dot(s, h);

    if (u < 0.0f || u > 1.0f)
      return hit;

    raym::vec3 q = raym::cross(s, edge1);
    float v = f * raym::dot(tDirection, q);

    if (v < 0.0f || u + v > 1.0f)
      return hit;

    float t = f * raym::dot(edge2, q);

    if (t > EPSILON)
    {
      hit.hit = true;
      hit.distance = t;
      hit.intersection = tOrigin + t * tDirection;

      if (smoothSmoothing)
      {
        float w = 1.0f - u - v;
        hit.normal = raym::normalize(
            w * normals[0] +
            u * normals[1] +
            v * normals[2]);
      }
      else
        hit.normal = raym::normalize(raym::cross(edge1, edge2));

      hit.intersection = raym::vec3(raym::to_vec3(transformationMatrix * raym::vec4(hit.intersection, 1.0)));
      hit.normal = raym::normalize(raym::vec3(raym::to_vec3(normalMatrix * raym::vec4(hit.normal, 0.0))));

      hit.distance = raym::length(hit.intersection - ray.origin);
      hit.object = this;
    }

    return hit;
  }

  void calculateBoundingBox()
  {
    boundingBox = AABB();
    for (const auto &vertex : vertices)
    {
      boundingBox.expand(vertex);
    }
  }
};
