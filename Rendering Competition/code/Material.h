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
struct Material{
    glm::vec3 ambient = glm::vec3(0.0);
    glm::vec3 diffuse = glm::vec3(1.0);
    glm::vec3 specular = glm::vec3(0.0);
    bool isDiffuse;
    float shininess = 0.0; //phong
    float alpha_x;        // Anisotropy in X
    float alpha_y;        // Anisotropy in Y
    bool useWardModel;    // Flag to use Ward Reflectance Model
};


#endif /* Material_h */
