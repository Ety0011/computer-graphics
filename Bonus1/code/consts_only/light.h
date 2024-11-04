#pragma once
#include <vector>

#include "../ray_math.h"
#include "object.h"

/**
 Light class
 */
class Light
{
public:
  raym::vec3 position; ///< Position of the light source
  raym::vec3 color;    ///< Color/intentisty of the light source
  ///
  Light(raym::vec3 position) : position(position)
  {
    color = raym::vec3(1.0);
  }

  Light(raym::vec3 position, raym::vec3 color) : position(position), color(color)
  {
  }
};

raym::vec3 ambient_light(0.001, 0.001, 0.001);

constexpr raym::vec3 PhongModel(const raym::vec3& point, const raym::vec3& normal, const raym::vec3& view_direction, const Material& material, std::vector<Light *> &lights) 
{
  raym::vec3 color(0.0);
  for (int light_num = 0; light_num < lights.size(); light_num++)
  {

    raym::vec3 light_direction = raym::normalize(lights[light_num]->position - point);
    raym::vec3 reflected_direction = raym::reflect(-1*light_direction, normal);// ! qua lo ricambio dopo non c'ho voglia

    float NdotL = raym::clamp(raym::dot(normal, light_direction), 0.0f, 1.0f);
    float VdotR = raym::clamp(raym::dot(view_direction, reflected_direction), 0.0f, 1.0f);

    raym::vec3 diffuse_color = material.diffuse;
    raym::vec3 diffuse = diffuse_color * raym::vec3(NdotL);
    raym::vec3 specular = material.specular * raym::vec3(raym::pow(VdotR, material.shininess));

    float r = raym::distance(point, lights[light_num]->position);
    r = raym::max(r, 0.1f);
    color = color + lights[light_num]->color * (diffuse + specular) / r / r;
  }
  color = color + (ambient_light * material.ambient);
  color = raym::clamp(color, raym::vec3(0.0), raym::vec3(1.0));

  return color;
}

inline raym::vec3 trace_ray(const Ray& ray, std::vector<Object *> &objects, std::vector<Light *> &lights)
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

  raym::vec3 color(0.0);
  if (closest_hit.hit)
    color = PhongModel(closest_hit.intersection, closest_hit.normal, raym::normalize(-1 * ray.direction), closest_hit.object->material, lights);
  else
    color = raym::vec3(0.0);

  return color;
}

constexpr raym::vec3 toneMapping(raym::vec3 intensity)
{
  float gamma = 1.0 / 2.0;
  float alpha = 12.0f;
  // ! occhio qua
  return raym::clamp(alpha * raym::pow(intensity, raym::vec3(gamma)), raym::vec3(0.0), raym::vec3(1.0));
}

inline std::vector<Light *> defineLights() {
  std::vector<Light *> lights;

  lights.push_back(new Light(raym::vec3(0, 26, 5), raym::vec3(1.0, 1.0, 1.0)));
	lights.push_back(new Light(raym::vec3(0, 1, 12), raym::vec3(0.1)));
	lights.push_back(new Light(raym::vec3(0, 5, 1), raym::vec3(0.4)));

  return lights;
}
