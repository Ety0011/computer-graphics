# Rendering Competition Report

**Authors**: Etienne Orio, Lino Candian

## Running the Project

To run our implementation:

1. Open the project in Visual Studio Code
2. Navigate to main.cpp
3. Click the "Run" button

The program will generate an output image named "result.ppm" in the project directory.

## Implemented Features

### Stochastic Ray Tracing

Key features of our stochastic implementation include:

**Anti-aliasing**: We reduce aliasing artifacts by casting multiple rays per pixel with random offsets. Each pixel is sampled multiple times, and the final color is computed as their average. This stochastic sampling helps break up the jagged edges that are typical in standard ray tracing.

**Soft Shadows**: Our soft shadow implementation simulates area lights instead of point lights. Each shadow computation samples random points on the light source, creating more realistic shadow transitions. The shadows are also restricted by a cone angle to control their spread and intensity.

**Depth of Field**: The depth of field effect creates realistic camera focus by simulating a lens of configurable size. Objects at the focal distance appear sharp, while objects closer or farther gradually blur. This is achieved by sampling different points on a virtual camera lens.

### Ward Reflectance Model

We implemented the Ward anisotropic reflectance model to handle materials with direction-dependent reflection properties. The implementation is integrated into our Phong shading model and activated when a material is marked as anisotropic.

### Additional Features

The following features are present in the source code but are currently deactivated:

- Photon Mapping
- Participating Media
