# Assignment 2: Transformations & Tone Mapping

**Authors**: Etienne Orio, Leonardo Asaro

## Solved Exercises

This submission includes solutions to the following exercises:

- **Exercise 1:** Ray-Plane Intersection.
- **Exercise 2:** Ray-Cone Intersection.
- **Exercise 3:** Light Attenuation and Tone Mapping.

## Encountered Problems

- **Exercise 1**: Ray-Plane Intersection  
    The derivation for the ray-plane intersection went smoothly, and the intersection calculations were relatively straightforward. An interesting part was how we handled the normal vector. Instead of recalculating the normal after the intersection, we simply took the plane's predefined normal and adjusted its direction based on the camera's position:

```cpp
    // Flip normal if it's pointing away from the camera
    glm::vec3 normal_facing_camera = normal;
    if (glm::dot(normal, ray.direction) > 0) {
        normal_facing_camera *= -1;
    }
```

- **Exercise 2**: Ray-Cone Intersection  
  This was by far the hardest exercise due to the numerous checks needed for proper handling of intersections. We struggled the most when adding the disk that closes cone because at some point we had huge black artifacts appearing on the scene. These artifacts were caused by an oversight in the code where we failed to check if the `planeHit.hit` value was actually `true` when returning `hit`:

```cpp
    if (tIntersectionPoint.y > 1) {
        ...
        // Missing !planeHit.hit in guard clause
        if (!planeHit.hit || pow(planeHit.intersection.x, 2.0f) + pow(planeHit.intersection.z, 2.0f) >= 1.0f) {
            return hit;
        }
        ...
        return hit;
    }
```

- **Exercise 3**: Light Attenuation and Tone Mapping  
  This was the easiest exercise to implement, but it turned out to be the hardest to configure. Initially, when all the material and lighting values were misconfigured, the rendered image was so dark that we thought we had made a mistake in the implementation. It took a lot of fine-tuning of parameters to get the lighting to look correct. This tuning process wasn't ideal, as it required a lot of trial and error and made the task feel more tedious than expected.
