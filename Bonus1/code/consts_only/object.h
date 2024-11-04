#pragma once
#include <array>

#include "material.h"
#include "../ray_math.h"

struct Ray
{
public:
  raym::vec3 origin;
  raym::vec3 direction;

  Ray(raym::vec3 origin, raym::vec3 direction) : origin(origin), direction(direction) {}
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

  virtual Hit intersect(const Ray& ray) = 0;

  constexpr void setTransformation(const raym::mat4& matrix)
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
  Plane(const raym::vec3& point, const raym::vec3& normal) : point(point), normal(normal) {}
  Plane(const raym::vec3& point, const raym::vec3& normal, const Material& material) : point(point), normal(normal)
  {
    this->material = material;
  }

  // ! constexpr fa si che questi calcoli li faccia il compilatore e non il programma
  constexpr float _DdotN(const Ray& ray) {
    return raym::dot(ray.direction, normal); 
  }

  constexpr float _PdotN(const Ray& ray) {
    return raym::dot(point - ray.origin, normal); 
  }

  // ! Uso le constexpr per PdotN e NdotN
  Hit intersect(const Ray& ray) override
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
  Plane *plane;
  std::array<raym::vec3, 3> vertices;
  std::array<raym::vec3, 3> normals;
  bool smoothSmoothing;

public:

  Triangle(Material material)
  {
    this->material = material;
    plane = new Plane(raym::vec3(0, 0, 0), raym::vec3(0, 0, -1));
    vertices[0] = raym::vec3(-0.5, -0.5, 0);
    vertices[1] = raym::vec3(0.5, -0.5, 0);
    vertices[2] = raym::vec3(0, 0.5, 0);
    smoothSmoothing = false;
  }

  constexpr raym::vec3 _triangle_normal() {
    return raym::normalize(raym::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
  }

  Triangle(const std::array<raym::vec3, 3>& vertices, const Material& material) : vertices(vertices)
  {
    this->material = material;
    raym::vec3 normal = _triangle_normal();
    plane = new Plane(vertices[0], normal); // ! Come possiamo evitare new?
    smoothSmoothing = false;
  }

  Triangle(const std::array<raym::vec3, 3>& vertices, const std::array<raym::vec3, 3>& normals, const Material& material) : vertices(vertices), normals(normals)
  {
    this->material = material;
    raym::vec3 normal = _triangle_normal();
    plane = new Plane(vertices[0], normal);
    smoothSmoothing = true;
  }

  // ! altre constexpr
  constexpr raym::vec3 _tOrigin(const Ray& ray) {
    return raym::to_vec3(inverseTransformationMatrix * raym::vec4(ray.origin, 1.0));
  }

  constexpr raym::vec3 _tDirection(const Ray& ray) {
    return raym::to_vec3(raym::normalize(inverseTransformationMatrix * raym::vec4(ray.direction, 0.0)));
  }

  constexpr raym::vec3 _Ncross() {
    return raym::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
  }

  constexpr raym::vec3 _ns(const int& i, const raym::vec3& p) {
    return raym::cross(vertices[i] - p, vertices[(i + 1) % 3] - p);
  }

  constexpr float _sign(const raym::vec3& N, const raym::vec3& n) {
    return raym::dot(N, n);
  }

  constexpr raym::vec3 _hitIntersection(const Hit& hit) {
    return raym::to_vec3(transformationMatrix * raym::vec4(hit.intersection, 1.0));
  }

  constexpr raym::vec3 _hitNormal(const Hit& hit) {
    return raym::to_vec3(raym::normalize((normalMatrix * raym::vec4(hit.normal, 0.0f))));
  }

  constexpr float _hitDistance(const Hit& hit, const Ray& ray) {
    return raym::length(hit.intersection - ray.origin);
  }

  Hit intersect(const Ray& ray)
  {

    Hit hit;
    hit.hit = false;

    raym::vec3 tOrigin = _tOrigin(ray);
    raym::vec3 tDirection = _tDirection(ray);

    Hit hitPlane = plane->intersect(Ray(tOrigin, tDirection));
    if (!hitPlane.hit)
    {
      return hit;
    }
    hit.intersection = hitPlane.intersection;
    hit.normal = hitPlane.normal;

    raym::vec3 p = hit.intersection;
    std::array<raym::vec3, 3> ns;

    raym::vec3 N = _Ncross();
    for (int i = 0; i < 3; i++)
    {
      ns[i] = _ns(i, p);
    }

    std::array<float, 3> signes;
    for (int i = 0; i < 3; i++)
    {
      float sign = _sign(N, ns[i]);
      if (sign < 0)
        return hit;
      signes[i] = sign;
    }

    std::array<float, 3> barycentrics;
    for (int i = 0; i < 3; i++)
    {
      barycentrics[i] = signes[i] / raym::pow(raym::length(N), 2);
    }

    if (smoothSmoothing)
    {
      hit.normal = raym::normalize(
          barycentrics[0] * normals[0] +
          barycentrics[1] * normals[1] +
          barycentrics[2] * normals[2]);
    }

    hit.hit = true;
    hit.object = this;
    hit.intersection = _hitIntersection(hit);
    hit.normal = _hitNormal(hit);
    hit.distance = raym::length(hit.intersection - ray.origin);

    return hit;
  }
};
