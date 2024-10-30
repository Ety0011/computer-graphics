# Assignment 3: Meshes

**Authors**: Etienne Orio, Leonardo Asaro

## Solved Exercises

This submission includes solutions to the following exercises:

- **Exercise 1**: Ray-Triangle Intersection and OBJ File Loader.
- **Exercise 2**: Smooth Normal Interpolation.
- **Bonus Exercise 3**: Custom Mesh Addition.

## Encountered Problems

- **Exercise 1**:
    We encountered an issue where, for some reason, the material of the last mesh added to the list changed the materials of all the other meshes, causing all objects to render with the same color even though they were supposed to be different. We identified the code responsible for this behavior and the solution was to modify the loop structure of this section of Mesh::intersect as follows:

    **Original Code**:

    ```cpp
    for (auto triangle : triangles) {
        Hit hit = triangle.intersect(ray);
        if (hit.hit && hit.distance < closest_hit.distance) {
            closest_hit = hit;
        }
    }
    ```

    **Fixed Code**:

    ```cpp
    for (int k = 0; k < triangles.size(); k++) {
        Hit hit = triangles[k].intersect(ray);
        if (hit.hit && hit.distance < closest_hit.distance) {
            closest_hit = hit;
        }
    }
    ```

    Although this change resolved the issue, we still don’t fully understand why the bug was occurring in the original loop. We would appreciate an explanation if possible, as understanding the underlying cause could help avoid similar issues in future projects.
