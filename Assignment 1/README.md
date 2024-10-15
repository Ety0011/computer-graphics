# Project: Ray Tracer

**Authors**: Leonardo Asaro, Etienne Orio

We successfully solved all the exercises provided in this project.

## 1. Solved Exercises Overview

- **Exercise 1**: Ray-Sphere Intersection  
  We implemented ray-sphere intersection by using the formulas we saw in class.  
  A mistake we made while implementing the change of origin was to translating also `ray.direction` in the new system of coordinates. Since `ray.direction` is a direction vector and should not be affected by translations, this caused artifacts in the rendering, showing circles instead of spheres.

- **Exercise 2**: Scene Definition  
  The sphere intersection worked out of the box and it produced the correct image.

- **Exercise 3**: Phong Lighting Model  
  Since we previously annotated to check cosines to be greater than zero and to normalize, the implementation of the Phong model has been smooth and chill.  
  Moreover, to make the end image more similar to the example in the exercise, we made a small change to the ambient lighting from `glm::vec3(0.1f) -> glm::vec3(1.0f)`.

## 2. Key Sections of Code

- **Ray-Sphere Intersection**

  ```cpp
  // Translate variables to new coordinate system  
    glm::vec3 translationOffset = ray.origin;  
        ray.origin -= translationOffset;  
        center -= translationOffset;  
    
        // Variable names are taken from notes  
    float a = glm::dot(center, ray.direction);  
        float d = std::sqrt(  
        std::pow(glm::length(center), 2) -  
        std::pow(a, 2)  
    );  
        float closest_t;  
    if (d == radius) {  
        closest_t = a;  
    } else if (d < radius) {  
        float b = std::sqrt(  
        std::pow(radius, 2) -  
        std::pow(d, 2)  
        );  
        float t1 = a - b;  
        float t2 = a + b;  
    
        // check if camera is not inside sphere  
        if (t1 < 0 && 0 < t2) {  
        closest_t = INFINITY;  
        } else {  
        t1 = t1 >= 0 ? t1 : INFINITY;  
        t2 = t2 >= 0 ? t2 : INFINITY;  
        closest_t = std::min(t1, t2);  
        }  
    } else {  
        closest_t = INFINITY;  
    }  
  
    // Translate back object variables to old coordinate system  
    ray.origin += translationOffset;  
    center += translationOffset;  
    
    if (closest_t != INFINITY) {  
        hit.hit = true;  
        hit.intersection = ray.direction * closest_t;  
    
        // Translate back intersection to old coordinate system  
        hit.intersection += ray.origin;  
    
        hit.distance = glm::distance(ray.origin, hit.intersection);  
        hit.normal = glm::normalize(hit.intersection - center);  
        hit.object = this;  
    }  
    return hit;
    ```

- **Phong Lighting Model**

    ```cpp
    // self-emitting intensity + ambient  
    color += material.ambient * ambient_light;  
    
    for (Light* light : lights) {  
        // diffuse  
        glm::vec3 light_direction = glm::normalize(light->position - point);  
        float cosOmega = glm::dot(normal, light_direction);  
        if (cosOmega > 0) {  
            color += material.diffuse * cosOmega * light->color;  
        }  
    
        // specular  
        glm::vec3 reflex_direction = glm::normalize(2.0f * normal * glm::dot(normal, light_direction) - light_direction);  
        float cosAlpha = glm::dot(view_direction, reflex_direction);  
        if (cosAlpha > 0) {  
            color += material.specular * glm::pow(cosAlpha, material.shininess) * light->color;  
        }  
    }  
    
    // The final color has to be clamped so the values do not go beyond 0 and 1.  
    color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0));  
    return color;
    ```
