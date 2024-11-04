#pragma once
#include "../glm/glm.hpp"
#include <array>
#include <vector>

#include "material.h"

struct Ray
{
public:
  glm::vec3 origin;
  glm::vec3 direction;

  Ray(glm::vec3 origin, glm::vec3 direction) : origin(origin), direction(direction) {}
};

class Object;

struct Hit
{
  bool hit;
  glm::vec3 normal;
  glm::vec3 intersection;
  float distance;
  Object *object;
};

class Object
{
protected:
  glm::mat4 transformationMatrix;
  glm::mat4 inverseTransformationMatrix;
  glm::mat4 normalMatrix;

public:
  glm::vec3 color;
  Material material;
  virtual Hit intersect(Ray ray) = 0;

  // Should be faster (?)
  void setTransformation(glm::mat4 matrix)
  {
    transformationMatrix = matrix;
    inverseTransformationMatrix = glm::inverse(matrix);
    normalMatrix = glm::transpose(inverseTransformationMatrix);
  }
};

class Plane : public Object
{

private:
  glm::vec3 normal;
  glm::vec3 point;

public:
  Plane(glm::vec3 point, glm::vec3 normal) : point(point), normal(normal)
  {
  }
  Plane(glm::vec3 point, glm::vec3 normal, Material material) : point(point), normal(normal)
  {
    this->material = material;
  }
  Hit intersect(Ray ray)
  {

    Hit hit;
    hit.hit = false;

    float DdotN = glm::dot(ray.direction, normal);
    if (DdotN < 0)
    {

      float PdotN = glm::dot(point - ray.origin, normal);
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
  std::array<glm::vec3, 3> vertices;
  std::array<glm::vec3, 3> normals;
  bool smoothSmoothing;

public:
  Triangle(Material material)
  {
    this->material = material;
    plane = new Plane(glm::vec3(0, 0, 0), glm::vec3(0, 0, -1));
    vertices[0] = glm::vec3(-0.5, -0.5, 0);
    vertices[1] = glm::vec3(0.5, -0.5, 0);
    vertices[2] = glm::vec3(0, 0.5, 0);
    smoothSmoothing = false;
  }
  Triangle(std::array<glm::vec3, 3> vertices, Material material) : vertices(vertices)
  {
    this->material = material;
    glm::vec3 normal = glm::normalize(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
    plane = new Plane(vertices[0], normal);
    smoothSmoothing = false;
  }
  Triangle(std::array<glm::vec3, 3> vertices, std::array<glm::vec3, 3> normals, Material material) : vertices(vertices), normals(normals)
  {
    this->material = material;
    glm::vec3 normal = glm::normalize(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
    plane = new Plane(vertices[0], normal);
    smoothSmoothing = true;
  }
  Hit intersect(Ray ray)
  {

    Hit hit;
    hit.hit = false;

    glm::vec3 tOrigin = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0);
    glm::vec3 tDirection = inverseTransformationMatrix * glm::vec4(ray.direction, 0.0);
    tDirection = glm::normalize(tDirection);

    Hit hitPlane = plane->intersect(Ray(tOrigin, tDirection));
    if (!hitPlane.hit)
    {
      return hit;
    }
    hit.intersection = hitPlane.intersection;
    hit.normal = hitPlane.normal;

    glm::vec3 p = hit.intersection;
    std::array<glm::vec3, 3> ns;

    glm::vec3 N = glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
    for (int i = 0; i < 3; i++)
    {
      ns[i] = glm::cross(vertices[i] - p, vertices[(i + 1) % 3] - p);
    }

    std::array<float, 3> signes;
    for (int i = 0; i < 3; i++)
    {
      float sign = glm::dot(N, ns[i]);
      if (sign < 0)
      {
        return hit;
      }
      signes[i] = sign;
    }

    std::array<float, 3> barycentrics;
    for (int i = 0; i < 3; i++)
    {
      barycentrics[i] = signes[i] / pow(glm::length(N), 2);
    }

    if (smoothSmoothing)
    {
      hit.normal = glm::normalize(
          barycentrics[0] * normals[0] +
          barycentrics[1] * normals[1] +
          barycentrics[2] * normals[2]);
    }

    hit.hit = true;
    hit.object = this;
    hit.intersection = transformationMatrix * glm::vec4(hit.intersection, 1.0);
    hit.normal = (normalMatrix * glm::vec4(hit.normal, 0.0));
    hit.normal = glm::normalize(hit.normal);
    hit.distance = glm::length(hit.intersection - ray.origin);

    return hit;
  }
};


