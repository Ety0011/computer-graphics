//
//  Material.h
//  Raytracer
//
//  Created by Piotr Didyk on 14.07.21.
//

#ifndef Material_h
#define Material_h

#include "glm/glm.hpp"

/**
 Structure describing a material of an object
 */
struct Material
{
    glm::vec3 ambient = glm::vec3(0.0);
    glm::vec3 diffuse = glm::vec3(1.0);
    glm::vec3 specular = glm::vec3(0.0);
    float shininess = 0.0f;

    bool is_reflective = false;
    float reflection = 0.0f;
    bool is_refractive = false;
    float refractive_index = 1.0f;

    bool is_anisotropic = false;
    float alpha_x = 0.1f;
    float alpha_y = 0.1f;
};

#endif /* Material_h */