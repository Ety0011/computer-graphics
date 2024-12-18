/**
@file main.cpp
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

#include <omp.h>
#include <iomanip>
#include <algorithm>
#include <random>

#include "Image.h"
#include "Material.h"

using namespace std;

/**
 Class representing a single ray.
 */
class Ray
{
public:
	glm::vec3 origin;	 ///< Origin of the ray
	glm::vec3 direction; ///< Direction of the ray
						 /**
						  Contructor of the ray
						  @param origin Origin of the ray
						  @param direction Direction of the ray
						  */
	Ray(glm::vec3 origin, glm::vec3 direction) : origin(origin), direction(direction)
	{
	}
};

class Object;

/**
 Structure representing the even of hitting an object
 */
struct Hit
{
	bool hit;				///< Boolean indicating whether there was or there was no intersection with an object
	glm::vec3 normal;		///< Normal vector of the intersected object at the intersection point
	glm::vec3 intersection; ///< Point of Intersection
	float distance;			///< Distance from the origin of the ray to the intersection point
	Object *object;			///< A pointer to the intersected object
	glm::vec3 tangent = glm::vec3(0.0f);
	glm::vec3 bitangent = glm::vec3(0.0f);
};

/**
 General class for the object
 */
class Object
{

protected:
	glm::mat4 transformationMatrix;		   ///< Matrix representing the transformation from the local to the global coordinate system
	glm::mat4 inverseTransformationMatrix; ///< Matrix representing the transformation from the global to the local coordinate system
	glm::mat4 normalMatrix;				   ///< Matrix for transforming normal vectors from the local to the global coordinate system

public:
	glm::vec3 color;   ///< Color of the object
	Material material; ///< Structure describing the material of the object
					   /** A function computing an intersection, which returns the structure Hit */
	virtual Hit intersect(Ray ray) = 0;

	/** Function that returns the material struct of the object*/
	Material getMaterial()
	{
		return material;
	}
	/** Function that set the material
	 @param material A structure describing the material of the object
	*/
	void setMaterial(Material material)
	{
		this->material = material;
	}
	/** Functions for setting up all the transformation matrices
	@param matrix The matrix representing the transformation of the object in the global coordinates */
	void setTransformation(glm::mat4 matrix)
	{

		transformationMatrix = matrix;

		inverseTransformationMatrix = glm::inverse(matrix);
		normalMatrix = glm::transpose(inverseTransformationMatrix);
	}
};

/**
 Implementation of the class Object for sphere shape.
 */
class Sphere : public Object
{
private:
	float radius;	  ///< Radius of the sphere
	glm::vec3 center; ///< Center of the sphere

public:
	/**
	 The constructor of the sphere
	 @param radius Radius of the sphere
	 @param center Center of the sphere
	 @param color Color of the sphere
	 */
	Sphere(float radius, glm::vec3 center, glm::vec3 color) : radius(radius), center(center)
	{
		this->color = color;
	}
	Sphere(float radius, glm::vec3 center, Material material) : radius(radius), center(center)
	{
		this->material = material;
	}
	/** Implementation of the intersection function*/
	Hit intersect(Ray ray)
	{

		glm::vec3 c = center - ray.origin;

		float cdotc = glm::dot(c, c);
		float cdotd = glm::dot(c, ray.direction);

		Hit hit;

		float D = 0;
		if (cdotc > cdotd * cdotd)
		{
			D = sqrt(cdotc - cdotd * cdotd);
		}
		if (D <= radius)
		{
			hit.hit = true;
			float t1 = cdotd - sqrt(radius * radius - D * D);
			float t2 = cdotd + sqrt(radius * radius - D * D);

			float t = t1;
			if (t < 0)
				t = t2;
			if (t < 0)
			{
				hit.hit = false;
				return hit;
			}

			hit.intersection = ray.origin + t * ray.direction;
			hit.normal = glm::normalize(hit.intersection - center);
			hit.distance = glm::distance(ray.origin, hit.intersection);
			hit.object = this;

			// Calculate tangent and bitangent vectors at the intersection point
			glm::vec3 normal = hit.normal;

			// Arbitrary vector (world up vector) to compute tangent
			glm::vec3 arbitrary_up = glm::vec3(0.0f, 1.0f, 0.0f);

			// If the normal is parallel to the arbitrary up vector, use a different vector
			if (glm::dot(normal, arbitrary_up) > 0.999f)
			{
				arbitrary_up = glm::vec3(1.0f, 0.0f, 0.0f);
			}

			// Calculate tangent vector (perpendicular to the normal)
			glm::vec3 tangent = glm::normalize(glm::cross(normal, arbitrary_up));
			// Calculate bitangent vector (perpendicular to both normal and tangent)
			glm::vec3 bitangent = glm::cross(normal, tangent);

			// Store tangent and bitangent in the hit object
			hit.tangent = tangent;
			hit.bitangent = bitangent;
		}
		else
		{
			hit.hit = false;
		}
		return hit;
	}
};

class Plane : public Object
{

private:
	glm::vec3 normal;
	glm::vec3 point;

public:
	Plane(glm::vec3 point, glm::vec3 normal) : point(point), normal(normal)
	{
	}
	Plane(glm::vec3 point, glm::vec3 normal, Material material) : point(point), normal(normal)
	{
		this->material = material;
	}
	Hit intersect(Ray ray)
	{

		Hit hit;
		hit.hit = false;

		float DdotN = glm::dot(ray.direction, normal);
		if (DdotN < 0)
		{

			float PdotN = glm::dot(point - ray.origin, normal);
			float t = PdotN / DdotN;

			if (t > 0)
			{
				hit.hit = true;
				hit.normal = normal;
				hit.distance = t;
				hit.object = this;
				hit.intersection = t * ray.direction + ray.origin;
			}
		}

		return hit;
	}
};

class Cone : public Object
{
private:
	Plane *plane;

public:
	Cone(Material material)
	{
		this->material = material;
		plane = new Plane(glm::vec3(0, 1, 0), glm::vec3(0.0, 1, 0));
	}
	Hit intersect(Ray ray)
	{

		Hit hit;
		hit.hit = false;

		glm::vec3 d = inverseTransformationMatrix * glm::vec4(ray.direction, 0.0); // implicit cast to vec3
		glm::vec3 o = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0);	   // implicit cast to vec3
		d = glm::normalize(d);

		float a = d.x * d.x + d.z * d.z - d.y * d.y;
		float b = 2 * (d.x * o.x + d.z * o.z - d.y * o.y);
		float c = o.x * o.x + o.z * o.z - o.y * o.y;

		float delta = b * b - 4 * a * c;

		if (delta < 0)
		{
			return hit;
		}

		float t1 = (-b - sqrt(delta)) / (2 * a);
		float t2 = (-b + sqrt(delta)) / (2 * a);

		float t = t1;
		hit.intersection = o + t * d;
		if (t < 0 || hit.intersection.y > 1 || hit.intersection.y < 0)
		{
			t = t2;
			hit.intersection = o + t * d;
			if (t < 0 || hit.intersection.y > 1 || hit.intersection.y < 0)
			{
				return hit;
			}
		};

		hit.normal = glm::vec3(hit.intersection.x, -hit.intersection.y, hit.intersection.z);
		hit.normal = glm::normalize(hit.normal);

		Ray new_ray(o, d);
		Hit hit_plane = plane->intersect(new_ray);
		if (hit_plane.hit && hit_plane.distance < t && length(hit_plane.intersection - glm::vec3(0, 1, 0)) <= 1.0)
		{
			hit.intersection = hit_plane.intersection;
			hit.normal = hit_plane.normal;
		}

		hit.hit = true;
		hit.object = this;
		hit.intersection = transformationMatrix * glm::vec4(hit.intersection, 1.0); // implicit cast to vec3
		hit.normal = (normalMatrix * glm::vec4(hit.normal, 0.0));					// implicit cast to vec3
		hit.normal = glm::normalize(hit.normal);
		hit.distance = glm::length(hit.intersection - ray.origin);

		return hit;
	}
};

/**
 Light class
 */
class Light
{
public:
	glm::vec3 position; ///< Position of the light source
	glm::vec3 color;	///< Color/intentisty of the light source
	float radius;
	Light(glm::vec3 position) : position(position)
	{
		color = glm::vec3(1.0);
		radius = 1.0;
	}
	Light(glm::vec3 position, glm::vec3 color) : position(position), color(color)
	{
		radius = 1.0;
	}
};

vector<Light *> lights; ///< A list of lights in the scene
// glm::vec3 ambient_light(0.1,0.1,0.1);
//  new ambient light
glm::vec3 ambient_light(0.0005, 0.0005, 0.0005);
vector<Object *> objects; ///< A list of all objects in the scene

float trace_shadow_ray(Ray shadow_ray, float light_distance)
{
	vector<Hit> hits;

	for (int k = 0; k < objects.size(); k++)
	{
		Hit hit = objects[k]->intersect(shadow_ray);
		if (hit.hit)
		{
			hits.push_back(hit);
		}
	}

	sort(hits.begin(), hits.end(), [](const Hit &a, const Hit &b)
		 { return a.distance < b.distance; });

	float shadow = 1.0f;

	for (const Hit &hit : hits)
	{
		if (hit.distance >= light_distance)
		{
			break;
		}

		Material material = hit.object->material;
		if (!material.is_refractive)
		{
			shadow = 0.0f;
			break;
		}
		else
		{
			shadow *= 0.9;
		}
	}

	return shadow;
}

glm::vec3 trace_ray(Ray ray);
glm::vec3 trace_ray_recursive(Ray ray, int depth_recursion);

glm::vec3 random_point_on_disk(float radius)
{
	// Generate random polar coordinates
	float theta = 2.0f * M_PI * float(rand()) / RAND_MAX; // Random angle [0, 2π]
	float r = radius * sqrt(float(rand()) / RAND_MAX);	  // Random radius with sqrt for uniform distribution

	// Convert polar to Cartesian coordinates
	float x = r * cos(theta);
	float z = r * sin(theta);

	// Disk lies on the XZ plane, Y is 0
	return glm::vec3(x, 0.0f, z);
}

glm::vec3 random_point_on_square(float light_size)
{
	float half_size = light_size * 0.5f;
	float x = (float(rand()) / RAND_MAX) * light_size - half_size;
	float z = (float(rand()) / RAND_MAX) * light_size - half_size;
	return glm::vec3(x, 0.0f, z); // Assuming the light is on the x-z plane
}

float compute_soft_shadow(const glm::vec3 &intersection_point, const glm::vec3 &light_position,
						  float light_radius, int shadow_samples)
{
	int unblocked_rays = 0; // Count of successful rays

	// Convert cone angle to cosine for comparison
	float cone_angle_deg = 90.0f;
	float cos_cone_angle = glm::cos(glm::radians(cone_angle_deg / 2.0f));

	for (int i = 0; i < shadow_samples; i++)
	{
		// 1. Generate a random point on the light source surface
		glm::vec3 random_point = light_position + random_point_on_square(light_radius);

		// 2. Create a shadow ray
		glm::vec3 light_direction = glm::normalize(random_point - intersection_point);

		// // 3. Check if light_direction is within the cone
		// glm::vec3 cone_direction = glm::vec3(0.0f, -1.0f, 0.0f); // Downward cone
		// if (glm::dot(-light_direction, cone_direction) > cos_cone_angle)
		// {
		// 	continue; // Skip rays outside the cone
		// }

		Ray shadow_ray(intersection_point + 1e-4f * light_direction, light_direction); // Avoid self-intersection

		// 4. Check for intersection with scene objects
		Hit closest_hit;
		closest_hit.hit = false;
		closest_hit.distance = INFINITY;

		for (int k = 0; k < objects.size(); k++)
		{
			Hit hit = objects[k]->intersect(shadow_ray);
			if (hit.hit && hit.distance < closest_hit.distance)
			{
				closest_hit = hit;
			}
		}

		// 5. If no object blocks the light, the ray is unblocked
		if (!closest_hit.hit || closest_hit.distance >= glm::distance(intersection_point, random_point))
		{
			unblocked_rays++;
		}
	}

	// 6. Compute visibility term
	return unblocked_rays / float(shadow_samples);
}

glm::vec3 calculate_ward_specular(glm::vec3 to_light_dir, glm::vec3 to_camera_dir, glm::vec3 normal,
								  glm::vec3 tangent, glm::vec3 bitangent, Material material)
{
	glm::vec3 halfVector = glm::normalize(to_light_dir + to_camera_dir);

	float NdotL = glm::dot(normal, to_light_dir);
	float NdotV = glm::dot(normal, to_camera_dir);

	if (NdotL < 0.0f || NdotV < 0.0f)
		return glm::vec3(0.0);

	float HdotN = glm::dot(halfVector, normal);
	float HdotT = glm::dot(halfVector, tangent);
	float HdotB = glm::dot(halfVector, bitangent);

	float exponent = -2 * (pow(HdotT / material.alpha_x, 2) + pow(HdotB / material.alpha_y, 2)) / (1 + HdotN);

	float ward_term = NdotL * exp(exponent) / (sqrt(NdotL * NdotV) * 4.0f * M_PI * material.alpha_x * material.alpha_y);

	return material.specular * ward_term;
}

/** Function for computing color of an object according to the Phong Model
 @param point A point belonging to the object for which the color is computer
 @param normal A normal vector the the point
 @param view_direction A normalized direction from the point to the viewer/camera
 @param material A material structure representing the material of the object
*/
glm::vec3 PhongModel(glm::vec3 point, glm::vec3 normal, glm::vec3 to_camera_dir, Material material, glm::vec3 tangent, glm::vec3 bitangent, int depth_recursion)
{
	glm::vec3 color(0.0);
	float epsilon = 1e-4f;
	int shadow_samples = 16;

	for (int light_num = 0; light_num < lights.size(); light_num++)
	{
		glm::vec3 to_light_dir = glm::normalize(lights[light_num]->position - point);
		glm::vec3 reflected_from_light_dir = glm::reflect(-to_light_dir, normal);
		float light_distance = glm::distance(point, lights[light_num]->position);

		float cosOmega = glm::clamp(glm::dot(normal, to_light_dir), 0.0f, 1.0f);
		glm::vec3 diffuse = material.diffuse * glm::vec3(cosOmega);

		glm::vec3 specular(0.0f);
		if (material.is_anisotropic)
		{
			// Works only with spheres
			specular = calculate_ward_specular(to_light_dir, to_camera_dir, normal, tangent, bitangent, material);
		}
		else
		{
			float cosAlpha = glm::clamp(glm::dot(to_camera_dir, reflected_from_light_dir), 0.0f, 1.0f);
			specular = material.specular * glm::vec3(pow(cosAlpha, material.shininess));
		}

		// Compute soft shadow visibility term
		float visibility = compute_soft_shadow(point, lights[light_num]->position, lights[light_num]->radius, shadow_samples);

		float r = max(light_distance, 0.1f);
		color += lights[light_num]->color * (diffuse + specular) * visibility / pow(r, 2.0f);
	}
	color += ambient_light * material.ambient;

	if (material.reflection > 0.0f)
	{
		glm::vec3 reflected_from_camera_dir = glm::reflect(-to_camera_dir, normal);
		Ray reflected_ray(point + epsilon * normal, reflected_from_camera_dir);
		glm::vec3 reflection_color = trace_ray_recursive(reflected_ray, depth_recursion + 1);
		color = color * (1 - material.reflection) + reflection_color * material.reflection;
	}
	if (material.is_refractive)
	{
		glm::vec3 n;
		float index1 = 1.0f;
		float index2 = 1.0f;
		if (glm::dot(-to_camera_dir, normal) < 0.0f)
		{
			index2 = material.refractive_index;
			n = normal;
		}
		else
		{
			index1 = material.refractive_index;
			n = -normal;
		}
		glm::vec3 refracted_form_camera_dir = glm::refract(-to_camera_dir, n, index1 / index2);
		Ray refracted_ray(point + epsilon * -n, refracted_form_camera_dir);
		glm::vec3 refraction_color = trace_ray_recursive(refracted_ray, depth_recursion + 1);
		color = refraction_color;
	}

	color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0));
	return color;
}

#define DEPTH_RECURSION_LIMIT 10

glm::vec3 trace_ray_recursive(Ray ray, int depth_recursion)
{
	glm::vec3 color(0.0);
	if (depth_recursion > DEPTH_RECURSION_LIMIT)
	{
		return color;
	}

	Hit closest_hit;

	closest_hit.hit = false;
	closest_hit.distance = INFINITY;

	for (int k = 0; k < objects.size(); k++)
	{
		Hit hit = objects[k]->intersect(ray);
		if (hit.hit == true && hit.distance < closest_hit.distance)
			closest_hit = hit;
	}

	if (closest_hit.hit)
	{
		color = PhongModel(closest_hit.intersection, closest_hit.normal, glm::normalize(-ray.direction), closest_hit.object->getMaterial(), closest_hit.tangent, closest_hit.bitangent, depth_recursion);
	}

	return color;
}

/**
 Functions that computes a color along the ray
 @param ray Ray that should be traced through the scene
 @return Color at the intersection point
 */
glm::vec3 trace_ray(Ray ray)
{
	return trace_ray_recursive(ray, 0);
}
/**
 Function defining the scene
 */
void sceneDefinition()
{
	float shininess_plane = 2.0f;
	float shininess_sphere = 10.0f;

	Material red_plane;
	red_plane.diffuse = glm::vec3(1.0f, 0.0f, 0.0f);
	red_plane.ambient = red_plane.diffuse / glm::vec3(10);
	red_plane.specular = glm::vec3(0.5);
	red_plane.shininess = shininess_plane;

	Material red_sphere;
	red_sphere.diffuse = glm::vec3(1.0f, 0.2f, 0.2f);
	red_sphere.ambient = red_sphere.diffuse / glm::vec3(10);
	red_sphere.specular = glm::vec3(0.5);
	red_sphere.shininess = shininess_sphere;
	red_sphere.is_anisotropic = true;
	red_sphere.alpha_y = 0.8f;

	Material blue_plane;
	blue_plane.diffuse = glm::vec3(0.0f, 0.0f, 1.0f);
	blue_plane.ambient = blue_plane.diffuse / glm::vec3(10);
	blue_plane.specular = glm::vec3(0.5);
	blue_plane.shininess = shininess_plane;

	Material blue_sphere;
	blue_sphere.diffuse = glm::vec3(0.2f, 0.2f, 1.0f);
	blue_sphere.ambient = blue_sphere.diffuse / glm::vec3(10);
	blue_sphere.specular = glm::vec3(0.5);
	blue_sphere.shininess = shininess_sphere;

	Material green_plane;
	green_plane.diffuse = glm::vec3(0.0f, 1.0f, 0.0f);
	green_plane.ambient = green_plane.diffuse / glm::vec3(10);
	green_plane.specular = glm::vec3(0.5);
	green_plane.shininess = shininess_plane;

	Material green_sphere;
	green_sphere.diffuse = glm::vec3(0.2f, 1.0f, 0.2f);
	green_sphere.ambient = green_sphere.diffuse / glm::vec3(10);
	green_sphere.specular = glm::vec3(0.5);
	green_sphere.shininess = shininess_sphere;
	green_sphere.is_anisotropic = true;
	green_sphere.alpha_x = 0.8f;

	// Spheres
	objects.push_back(new Sphere(1.0, glm::vec3(-2, -1, 5), green_sphere));
	objects.push_back(new Sphere(0.5, glm::vec3(0, -2.5, 4), blue_sphere));
	objects.push_back(new Sphere(0.5, glm::vec3(1.5, -2.5, 3), red_sphere));

	// Lights
	lights.push_back(new Light(glm::vec3(0, 2.99, 4), glm::vec3(0.1)));

	// Planes
	// planes above and below
	objects.push_back(new Plane(glm::vec3(0, -3, 0), glm::vec3(0.0, 1, 0)));
	objects.push_back(new Plane(glm::vec3(0, 3, 0), glm::vec3(0.0, -1, 0)));

	// planes right and left
	objects.push_back(new Plane(glm::vec3(-3, 0, 0), glm::vec3(1.0, 0.0, 0.0), red_plane));
	objects.push_back(new Plane(glm::vec3(3, 0, 0), glm::vec3(-1.0, 0.0, 0.0), green_plane));

	// plane front
	objects.push_back(new Plane(glm::vec3(0, 0, 6), glm::vec3(0.0, 0.0, -1.0)));
}
glm::vec3 toneMapping(glm::vec3 intensity)
{
	float gamma = 1.0 / 2.0;
	float alpha = 12.0f;
	return glm::clamp(alpha * glm::pow(intensity, glm::vec3(gamma)), glm::vec3(0.0), glm::vec3(1.0));
}

void printProgress(float progress, float eta_seconds)
{
	int barWidth = 70;
	cout << "[";
	int pos = barWidth * progress;
	for (int i = 0; i < barWidth; ++i)
	{
		if (i < pos)
			cout << "=";
		else if (i == pos)
			cout << ">";
		else
			cout << " ";
	}
	cout << "] " << int(progress * 100.0) << " %";
	cout << " ETA: " << setw(4) << fixed << setprecision(1) << eta_seconds << "s  \r";
	cout.flush();
}

glm::vec2 randomPointOnDisk(float radius)
{
	static std::default_random_engine generator;
	static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

	float r = radius * sqrt(distribution(generator));	 // Disk radius
	float theta = 2.0f * M_PI * distribution(generator); // Angle
	return glm::vec2(r * cos(theta), r * sin(theta));
}

glm::vec3 depthOfFieldRayTrace(const Ray &primary_ray,
							   float aperture_radius, float focal_length, int dof_samples)
{
	glm::vec3 color(0.0f);

	for (int i = 0; i < dof_samples; i++)
	{
		// 1. Sample a point on the aperture (lens)
		glm::vec2 lens_sample = randomPointOnDisk(aperture_radius);
		glm::vec3 lens_point = primary_ray.origin + glm::vec3(lens_sample.x, lens_sample.y, 0.0f);

		// 2. Compute the focal point
		float t = focal_length / glm::dot(primary_ray.direction, glm::vec3(0, 0, 1)); // Assumes camera looks along Z
		glm::vec3 focal_point = primary_ray.origin + t * primary_ray.direction;

		// 3. Create a new ray toward the focal point
		glm::vec3 new_direction = glm::normalize(focal_point - lens_point);
		Ray new_ray(lens_point, new_direction);

		// 4. Trace the new ray and accumulate color
		color += trace_ray(new_ray);
	}

	// 5. Average the color
	return color / float(dof_samples);
}

int main(int argc, const char *argv[])
{
	omp_set_num_threads(12);

	clock_t t = clock(); // variable for keeping the time of the rendering
	clock_t start_time = clock();

	int width = 500;  // width of the image
	int height = 500; // height of the image
	float fov = 90;	  // field of view

	sceneDefinition(); // Let's define a scene

	Image image(width, height); // Create an image where we will store the result
	vector<glm::vec3> image_values(width * height);

	float s = 2 * tan(0.5 * fov / 180 * M_PI) / width;
	float X = -s * width / 2;
	float Y = s * height / 2;

	int totalPixels = width * height;
	int iteration = 0;

	int aa_samples = 10;  // Supersampling for anti-aliasing
	int dof_samples = 10; // Number of samples for depth of field
	// DEFAULT 0.5f
	float aperture_radius = 0.05f; // Controls the size of the blur (lens aperture)
	// DEFAULT 8.0f
	float focal_length = 3.0f; // Distance to the focal plane

#pragma omp parallel for
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
		{
			glm::vec3 color(0.0f); // Accumulator for averaged color

			for (int k = 0; k < aa_samples; ++k)
			{
				float x_offset = rand() * (1.0f / RAND_MAX);
				float y_offset = rand() * (1.0f / RAND_MAX);

				float dx = X + (i + x_offset) * s + s / 2;
				float dy = Y - (j + y_offset) * s - s / 2;
				float dz = 1;

				glm::vec3 origin(0, 0, 0);
				glm::vec3 direction(dx, dy, dz);
				direction = glm::normalize(direction);
				Ray ray(origin, direction);

				color += depthOfFieldRayTrace(ray, aperture_radius, focal_length, dof_samples);
			}

			color /= aa_samples;

			image.setPixel(i, j, toneMapping(color));

			if (iteration % (totalPixels / 100) == 0)
			{
#pragma omp critical
				{
					// Calculate progress
					float progress = (float)(iteration) / totalPixels;

					// Calculate elapsed time
					clock_t current_time = clock();
					float elapsed_seconds = float(current_time - start_time) / CLOCKS_PER_SEC;

					// Estimate remaining time
					float eta_seconds = (elapsed_seconds / progress) * (1 - progress);

					printProgress(progress, eta_seconds);
				}
			}
			iteration++;
		}

	cout << endl;
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