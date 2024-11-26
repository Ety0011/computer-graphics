//
//  Material.h
//  Raytracer
//
//  Created by Piotr Didyk on 14.07.21.
//

#ifndef Material_h
#define Material_h

#include "../ray_math.h"

/**
 Structure describing a material of an object
 */
struct Material{
  raym::vec3 ambient = raym::vec3(0.0);
  raym::vec3 diffuse = raym::vec3(1.0);
  raym::vec3 specular = raym::vec3(0.0);
  float shininess = 0.0;
};

#endif /* Material_h */
