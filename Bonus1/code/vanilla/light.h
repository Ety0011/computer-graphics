#pragma once
#include "../glm/glm.hpp"
#include <vector>

#include "object.h"

/**
 Light class
 */
class Light
{
public:
  glm::vec3 position; ///< Position of the light source
  glm::vec3 color;    ///< Color/intentisty of the light source
  Light(glm::vec3 position) : position(position)
  {
    color = glm::vec3(1.0);
  }
  Light(glm::vec3 position, glm::vec3 color) : position(position), color(color)
  {
  }
};

glm::vec3 ambient_light(0.001, 0.001, 0.001);

glm::vec3 PhongModel(glm::vec3 point, glm::vec3 normal, glm::vec3 view_direction, Material material, std::vector<Light *> &lights)
{

  glm::vec3 color(0.0);
  for (int light_num = 0; light_num < lights.size(); light_num++)
  {

    glm::vec3 light_direction = glm::normalize(lights[light_num]->position - point);
    glm::vec3 reflected_direction = glm::reflect(-light_direction, normal);

    float NdotL = glm::clamp(glm::dot(normal, light_direction), 0.0f, 1.0f);
    float VdotR = glm::clamp(glm::dot(view_direction, reflected_direction), 0.0f, 1.0f);

    glm::vec3 diffuse_color = material.diffuse;
    glm::vec3 diffuse = diffuse_color * glm::vec3(NdotL);
    glm::vec3 specular = material.specular * glm::vec3(pow(VdotR, material.shininess));

    float r = glm::distance(point, lights[light_num]->position);
    r = glm::max(r, 0.1f);
    color += lights[light_num]->color * (diffuse + specular) / r / r;
  }
  color += ambient_light * material.ambient;
  color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0));
  return color;
}

glm::vec3 trace_ray(Ray ray, std::vector<Object *> &objects, std::vector<Light *> &lights)
{

  Hit closest_hit;

  closest_hit.hit = false;
  closest_hit.distance = INFINITY;

  for (int k = 0; k < objects.size(); k++)
  {
    Hit hit = objects[k]->intersect(ray);
    if (hit.hit == true && hit.distance < closest_hit.distance)
      closest_hit = hit;
  }

  glm::vec3 color(0.0);
  if (closest_hit.hit)
    color = PhongModel(closest_hit.intersection, closest_hit.normal, glm::normalize(-ray.direction), closest_hit.object->material, lights);
  else
    color = glm::vec3(0.0, 0.0, 0.0);

  return color;
}

glm::vec3 toneMapping(glm::vec3 intensity)
{
  float gamma = 1.0 / 2.0;
  float alpha = 12.0f;
  return glm::clamp(alpha * glm::pow(intensity, glm::vec3(gamma)), glm::vec3(0.0), glm::vec3(1.0));
}

std::vector<Light *> defineLights() {
  std::vector<Light *> lights;

  lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(1.0, 1.0, 1.0)));
	lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.1)));
	lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4)));

  return lights;
}
