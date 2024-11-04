/**
@file main.cpp
*/

#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>

using namespace std;

// PROFILE DEFINITION
#ifndef PROFILE_NAME
#define PROFILE_NAME vanilla
#endif

#define EXPAND_AND_CONCAT(a, b) EXPAND_AND_CONCAT_IMPL(a, b)
#define EXPAND_AND_CONCAT_IMPL(a, b) a##b

#define USE_GLM(profile) EXPAND_AND_CONCAT(IS_VANILLA_, profile)

#define IS_VANILLA_vanilla 1
#define IS_VANILLA_consts_only 0

#define STRING(x) #x
#define INCLUDE_FILE(profile, file) STRING(profile/file.h)

#include INCLUDE_FILE(PROFILE_NAME, object)
#include INCLUDE_FILE(PROFILE_NAME, mesh)
#include INCLUDE_FILE(PROFILE_NAME, light)
#include INCLUDE_FILE(PROFILE_NAME, material)
#include INCLUDE_FILE(PROFILE_NAME, image)

#if USE_GLM(PROFILE_NAME)
#define GLM 1
#else
#define GLM 0
#endif

#if GLM
#include "glm/glm.hpp"
#include "glm/geometric.hpp"

namespace math {
  using vec3 = glm::vec3;
  using vec4 = glm::vec4;
  using mat4 = glm::mat4;

  // Functions
  inline glm::vec3 normalize(const vec3& v) {
    return glm::normalize(v);
  }

}
#else
#include "ray_math.h"

namespace math {
  // Type aliases
  using vec3 = raym::vec3;
  using vec4 = raym::vec4;
  using mat4 = raym::mat4;

  // Functions
  constexpr raym::vec3 normalize(const vec3& v) {
    return raym::normalize(v);
  }
}
#endif

std::vector<Light *> lights = defineLights();
std::vector<Object *> objects = defineObjects();

void printProgress(float percentage)
{
	int barWidth = 70; // Width of the progress bar

	std::cout << "[";
	int pos = barWidth * percentage;
	for (int i = 0; i < barWidth; ++i)
	{
		if (i < pos)
			std::cout << "▮";
		else
			std::cout << ".";
	}
	std::cout << "] " << int(percentage * 100.0) << " %\r";
	std::cout.flush();
}

int main(int argc, const char *argv[])
{
	clock_t t = clock(); // variable for keeping the time of the rendering

	int width = 480;	
	int height = 240; 
	float fov = 90;		

	Image image(width, height); // Create an image where we will store the result
	vector<math::vec3> image_values(width * height);

	float s = 2 * tan(0.5 * fov / 180 * M_PI) / width;
	float X = -s * width / 2;
	float Y = s * height / 2;

	int totalPixels = width * height;
	int processed = 0;

	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{

			float dx = X + i * s + s / 2;
			float dy = Y - j * s - s / 2;
			float dz = 1;

			math::vec3 origin(0, 0, 0);
			math::vec3 direction(dx, dy, dz);
			direction = math::normalize(direction);

			Ray ray(origin, direction);
			image.setPixel(i, j, toneMapping(trace_ray(ray, objects, lights)));

			processed++;
			if (processed % (totalPixels / 100) == 0)
				printProgress((float)processed / totalPixels);
		}

	std::cout << std::endl;

	t = clock() - t;
	cout << "It took " << ((float)t) / CLOCKS_PER_SEC << " seconds to render the image." << endl;
	cout << "I could render at " << (float)CLOCKS_PER_SEC / ((float)t) << " frames per second." << endl;

	// Writing the final results of the rendering
	if (argc == 2)
	{
		image.writeImage(argv[1]);
	}
	else
	{
		image.writeImage("./result.ppm");
	}

	return 0;
}
