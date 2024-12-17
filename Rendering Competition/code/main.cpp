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
#include <array>
#include <sstream>
#include <algorithm>
#include <random>

#include "Image.h"
#include "Material.h"

using namespace std;

// our code
// btw -> see header -> TODO: ask prof is we need to implemnt it here
// just easier to send the assignmend
//  the one in the header was commented

/**
 Structure describing a material of an object
 */

struct Material
{
	glm::vec3 ambient = glm::vec3(0.0);
	;
	glm::vec3 diffuse = glm::vec3(1.0);
	glm::vec3 specular = glm::vec3(0.0);

	float shininess = 0.0; // for Phong model

	// add the below fields
	bool refract_flag = false; // Indicate if is refractive

	// reflect and rerefractivenes
	float reflex = 0.0;		 // how much reflectiveness it is
	float refract_idx = 1.0; // represent refractiveness

	float sigmaT; // Tangential anisotropy factor
	float sigmaB; // Bitangential anisotropy factor
};

// our code end

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
	virtual void setTransformation(glm::mat4 matrix)
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

class AABB
{
public:
	glm::vec3 minBounds;
	glm::vec3 maxBounds;

	AABB() : minBounds(glm::vec3(FLT_MAX)), maxBounds(glm::vec3(-FLT_MAX)) {}
	AABB(const glm::vec3 &min, const glm::vec3 &max) : minBounds(min), maxBounds(max) {}

	void expand(const glm::vec3 &point)
	{
		minBounds = glm::min(minBounds, point);
		maxBounds = glm::max(maxBounds, point);
	}

	void merge(const AABB &other)
	{
		minBounds = glm::min(minBounds, other.minBounds);
		maxBounds = glm::max(maxBounds, other.maxBounds);
	}

	bool intersect(const Ray &ray)
	{
		glm::vec3 tmin = (minBounds - ray.origin) / ray.direction;
		glm::vec3 tmax = (maxBounds - ray.origin) / ray.direction;

		glm::vec3 tNear = glm::min(tmin, tmax);
		glm::vec3 tFar = glm::max(tmin, tmax);

		float tXmin = tNear.x;
		float tXmax = tFar.x;
		float tYmin = tNear.y;
		float tYmax = tFar.y;
		float tZmin = tNear.z;
		float tZmax = tFar.z;

		int overlapCount = 0;

		if (tXmin <= tYmax && tXmax >= tYmin)
		{
			overlapCount++;
		}

		if (tXmin <= tZmax && tXmax >= tZmin)
		{
			overlapCount++;
		}

		if (tYmin <= tZmax && tYmax >= tZmin)
		{
			overlapCount++;
		}

		return overlapCount >= 2;
	}

	float distanceTo(const Ray &ray)
	{
		glm::vec3 tmin = (minBounds - ray.origin) / ray.direction;
		glm::vec3 tmax = (maxBounds - ray.origin) / ray.direction;

		glm::vec3 tNear = glm::min(tmin, tmax);
		glm::vec3 tFar = glm::max(tmin, tmax);

		float tXmin = tNear.x;
		float tXmax = tFar.x;
		float tYmin = tNear.y;
		float tYmax = tFar.y;
		float tZmin = tNear.z;
		float tZmax = tFar.z;

		if (tXmax < 0 || tYmax < 0 || tZmax < 0)
		{
			return INFINITY;
		}

		return max(0.0f, max(tXmin, max(tYmin, tZmin)));
	}
};

class Triangle : public Object
{
private:
	Plane *plane;
	array<glm::vec3, 3> vertices;
	array<glm::vec3, 3> normals;
	bool smoothSmoothing;

public:
	AABB boundingBox;
	Triangle(array<glm::vec3, 3> vertices, Material material) : vertices(vertices)
	{
		this->material = material;
		glm::vec3 normal = glm::normalize(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
		plane = new Plane(vertices[0], normal);
		smoothSmoothing = false;
		calculateBoundingBox();
	}
	Triangle(array<glm::vec3, 3> vertices, array<glm::vec3, 3> normals, Material material) : vertices(vertices), normals(normals)
	{
		this->material = material;
		glm::vec3 normal = glm::normalize(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
		plane = new Plane(vertices[0], normal);
		smoothSmoothing = true;
		calculateBoundingBox();
	}
	Hit intersect(Ray ray)
	{

		Hit hit;
		hit.hit = false;

		glm::vec3 tOrigin = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0);
		glm::vec3 tDirection = inverseTransformationMatrix * glm::vec4(ray.direction, 0.0);
		tDirection = glm::normalize(tDirection);

		Hit hitPlane = plane->intersect(Ray(tOrigin, tDirection));
		if (!hitPlane.hit)
		{
			return hit;
		}
		hit.intersection = hitPlane.intersection;
		hit.normal = hitPlane.normal;

		glm::vec3 p = hit.intersection;
		array<glm::vec3, 3> ns;
		glm::vec3 N = glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0]);
		for (int i = 0; i < 3; i++)
		{
			ns[i] = glm::cross(vertices[(i + 1) % 3] - p, vertices[(i + 2) % 3] - p);
		}

		array<float, 3> signes;
		for (int i = 0; i < 3; i++)
		{
			float sign = glm::dot(N, ns[i]);
			if (sign < 0)
			{
				return hit;
			}
			signes[i] = sign;
		}

		array<float, 3> barycentrics;
		for (int i = 0; i < 3; i++)
		{
			barycentrics[i] = signes[i] / pow(glm::length(N), 2);
		}

		if (smoothSmoothing)
		{
			hit.normal = glm::normalize(
				barycentrics[0] * normals[0] +
				barycentrics[1] * normals[1] +
				barycentrics[2] * normals[2]);
		}

		hit.hit = true;
		hit.object = this;
		hit.intersection = transformationMatrix * glm::vec4(hit.intersection, 1.0);
		hit.normal = (normalMatrix * glm::vec4(hit.normal, 0.0));
		hit.normal = glm::normalize(hit.normal);
		hit.distance = glm::length(hit.intersection - ray.origin);

		return hit;
	}

	void calculateBoundingBox()
	{
		boundingBox = AABB();
		for (const auto &vertex : vertices)
		{
			boundingBox.expand(vertex);
		}
	}
};

struct BVHNode
{
	AABB boundingBox;
	BVHNode *left = nullptr;
	BVHNode *right = nullptr;
	vector<Triangle *> triangles;

	bool isLeaf() const { return left == nullptr && right == nullptr; }
};

class Mesh : public Object
{
private:
	vector<Triangle *> triangles;
	BVHNode *bvhRoot;
	int smoothShading;

public:
	Mesh(const string &filename)
	{
		smoothShading = 0;
		if (!loadFromFile(filename))
		{
			throw runtime_error("Failed to load mesh from file: " + filename);
		}
		bvhRoot = buildBVH(triangles);
	}
	Mesh(const string &filename, Material material)
	{
		this->material = material;
		smoothShading = 0;
		if (!loadFromFile(filename))
		{
			throw runtime_error("Failed to load mesh from file: " + filename);
		}
		bvhRoot = buildBVH(triangles);
	}

	void setTransformation(glm::mat4 matrix) override
	{
		transformationMatrix = matrix;
		inverseTransformationMatrix = glm::inverse(matrix);
		normalMatrix = glm::transpose(inverseTransformationMatrix);
		setTransformationTriangles(matrix);
	}

	void setTransformationTriangles(glm::mat4 matrix)
	{
		for (Triangle *triangle : triangles)
		{
			triangle->setTransformation(matrix);
		}
	}

	bool loadFromFile(const string &filename)
	{
		ifstream file(filename);
		if (!file.is_open())
		{
			cerr << "Failed to open file: " << filename << endl;
			return false;
		}

		vector<glm::vec3> vertices;
		vector<glm::vec3> normals;
		string line;
		while (getline(file, line))
		{
			istringstream lineStream(line);
			string prefix;
			lineStream >> prefix;

			if (prefix == "s")
			{
				bool x;
				lineStream >> x;
				smoothShading = x;
			}
			else if (prefix == "v")
			{
				float x, y, z;
				lineStream >> x >> y >> z;
				vertices.emplace_back(x, y, z);
			}
			else if (prefix == "vn")
			{
				float nx, ny, nz;
				lineStream >> nx >> ny >> nz;
				normals.emplace_back(nx, ny, nz);
			}
			else if (prefix == "f")
			{
				vector<int> vIndices;
				vector<int> vnIndices;
				string indexString;

				while (lineStream >> indexString)
				{
					istringstream indexStream(indexString);
					int vIndex, vtIndex, vnIndex;

					indexStream >> vIndex;
					vIndices.push_back(vIndex - 1);

					if (indexStream.peek() != '/')
					{
						continue;
					}
					indexStream.ignore();

					if (indexStream.peek() != '/')
					{
						indexStream >> vtIndex;
					}

					if (indexStream.peek() == '/')
					{
						indexStream.ignore();
						indexStream >> vnIndex;
						vnIndices.push_back(vnIndex - 1);
					}
				}

				array<glm::vec3, 3> triangleVertices;
				for (int i = 0; i < 3; i++)
				{
					triangleVertices[i] = vertices[vIndices[i]];
				}

				Triangle *triangle = new Triangle(triangleVertices, this->material);
				if (smoothShading == 1)
				{
					array<glm::vec3, 3> triangleNormals;
					for (int i = 0; i < 3; i++)
					{
						triangleNormals[i] = normals[vnIndices[i]];
					}
					triangle = new Triangle(triangleVertices, triangleNormals, this->material);
				}

				triangles.push_back(triangle);
			}
		}
		file.close();
		return true;
	}

	Hit intersect(Ray ray)
	{
		return traverseBVH(bvhRoot, ray);
	}

	BVHNode *buildBVH(vector<Triangle *> &triangles)
	{
		BVHNode *node = new BVHNode();

		for (Triangle *triangle : triangles)
		{
			node->boundingBox.merge(triangle->boundingBox);
		}

		if (triangles.size() <= 3)
		{
			node->triangles = triangles;
			return node;
		}

		glm::vec3 extent = node->boundingBox.maxBounds - node->boundingBox.minBounds;
		int axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2);

		sort(
			triangles.begin(),
			triangles.end(),
			[axis](Triangle *a, Triangle *b)
			{
				return a->boundingBox.minBounds[axis] < b->boundingBox.minBounds[axis];
			});

		int mid = triangles.size() / 2;
		vector<Triangle *> left(triangles.begin(), triangles.begin() + mid);
		vector<Triangle *> right(triangles.begin() + mid, triangles.end());

		node->left = buildBVH(left);
		node->right = buildBVH(right);

		return node;
	}

	Hit traverseBVH(BVHNode *node, const Ray &ray)
	{
		Hit closestHit;
		closestHit.hit = false;
		closestHit.distance = INFINITY;

		glm::vec3 tOrigin = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0);
		glm::vec3 tDirection = inverseTransformationMatrix * glm::vec4(ray.direction, 0.0);
		tDirection = glm::normalize(tDirection);
		Ray new_ray = Ray(tOrigin, tDirection);

		if (!node->boundingBox.intersect(new_ray))
		{
			return closestHit;
		}

		if (node->isLeaf())
		{
			// TODO: should use function closest_hit(node->triangles)
			for (Triangle *object : node->triangles)
			{
				Hit hit = object->intersect(ray);
				if (hit.hit && hit.distance < closestHit.distance)
				{
					closestHit = hit;
				}
			}
			return closestHit;
		}

		Hit leftHit = traverseBVH(node->left, ray);
		Hit rightHit = traverseBVH(node->right, ray);

		if (leftHit.hit && leftHit.distance < closestHit.distance)
		{
			closestHit = leftHit;
		}
		if (rightHit.hit && rightHit.distance < closestHit.distance)
		{
			closestHit = rightHit;
		}

		return closestHit;
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
	Light(glm::vec3 position) : position(position)
	{
		color = glm::vec3(1.0);
	}
	Light(glm::vec3 position, glm::vec3 color) : position(position), color(color)
	{
	}
};

vector<Light *> lights; ///< A list of lights in the scene
// glm::vec3 ambient_light(0.1,0.1,0.1);
//  new ambient light
glm::vec3 ambient_light(0.001, 0.001, 0.001);
vector<Object *> objects; ///< A list of all objects in the scene

// our code starts
// jumps between one object and another
int MAX_JUMP = 5;
int jump = 0;

// tollerance value
// to high = noise
float tol = 0.001f;

// lecture8, slide 38
// ask TA if this is ok? add a new function?
float compute_fresnel(float delta_1, float delta_2, float cos_1, float cos_2)
{
	float part_1 = (delta_1 * cos_1 - delta_2 * cos_2) / (delta_1 * cos_1 + delta_2 * cos_2);
	part_1 *= part_1;
	float part_2 = (delta_2 * cos_1 - delta_1 * cos_2) / (delta_2 * cos_1 + delta_1 * cos_2);
	part_2 *= part_2;
	return 0.5f * (part_1 + part_2);
}
// our code ends
float rand_float()
{
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

glm::vec3 WardSpecular(glm::vec3 light_direction, glm::vec3 view_direction, glm::vec3 normal, glm::vec3 tangent, glm::vec3 bitangent, float sigmaT, float sigmaB)
{
	glm::vec3 light_tangent = glm::dot(light_direction, tangent) * tangent;
	glm::vec3 light_bitangent = glm::dot(light_direction, bitangent) * bitangent;
	glm::vec3 view_tangent = glm::dot(view_direction, tangent) * tangent;
	glm::vec3 view_bitangent = glm::dot(view_direction, bitangent) * bitangent;

	float lT = glm::length(light_tangent - view_tangent);
	float lB = glm::length(light_bitangent - view_bitangent);

	float exponent = -(lT * lT / (2.0f * sigmaT * sigmaT) + lB * lB / (2.0f * sigmaB * sigmaB));
	glm::vec3 specular = glm::vec3(exp(exponent) / (M_PI * sigmaT * sigmaB)) * glm::vec3(1.0f);

	return specular;
}

/** Function for computing color of an object according to the Phong Model
 @param point A point belonging to the object for which the color is computer
 @param normal A normal vector the the point
 @param view_direction A normalized direction from the point to the viewer/camera
 @param material A material structure representing the material of the object
*/
glm::vec3 PhongModel(glm::vec3 point, glm::vec3 normal, glm::vec3 view_direction, Material material)
{
	glm::vec3 color(0.0);
	for (int light_num = 0; light_num < lights.size(); light_num++)
	{

		glm::vec3 light_direction = glm::normalize(lights[light_num]->position - point);

		// Start shadow calculation
		Ray shadow_ray(point + normal * 0.01f, light_direction); // Small offset to avoid self-shadowing
		float light_distance = glm::distance(point, lights[light_num]->position);

		bool in_shadow = false;
		for (int obj_num = 0; obj_num < objects.size(); obj_num++)
		{
			// Compute shadow
			Hit shadow_hit = objects[obj_num]->intersect(shadow_ray);
			if (shadow_hit.hit && shadow_hit.distance > tol && shadow_hit.distance < light_distance - tol)
			{
				in_shadow = true;
				break;
			}
		}

		if (!in_shadow)
		{
			// Phong diffuse component (same as before)
			glm::vec3 reflected_direction = glm::reflect(-light_direction, normal);
			float NdotL = glm::clamp(glm::dot(normal, light_direction), 0.0f, 1.0f);
			glm::vec3 diffuse_color = material.diffuse;
			glm::vec3 diffuse = diffuse_color * glm::vec3(NdotL);

			// Ward specular component (replace Phong specular)
			glm::vec3 tangent = glm::normalize(glm::cross(normal, glm::vec3(0.0f, 1.0f, 0.0f))); // Simplified, should use actual tangent
			glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));					 // Simplified, should use actual bitangent
			glm::vec3 specular = WardSpecular(light_direction, view_direction, normal, tangent, bitangent, material.sigmaT, material.sigmaB);

			// Combine diffuse and specular, consider attenuation
			float r = glm::distance(point, lights[light_num]->position);
			r = glm::max(r, 0.1f); // Prevent division by zero
			color += lights[light_num]->color * (diffuse + specular) / (r * r);
		}
	}

	color += ambient_light * material.ambient;				   // Add ambient component
	color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0)); // Clamp the color to [0, 1]
	return color;
}

// Simplex or Perlin Noise implementation
float perlin_noise(glm::vec2 position)
{
	// This is a very simple implementation. A more complex noise function would use a gradient table
	// and fade functions to interpolate smoothly. This is just for demonstration.

	int xi = static_cast<int>(floor(position.x)) & 255;
	int yi = static_cast<int>(floor(position.y)) & 255;

	// Using a simple hash function to generate random values
	float random_val = (sin(xi + yi * 57.0f) * 43758.5453f);
	return (random_val - floor(random_val));
}

/**
 Functions that computes a color along the ray
 @param ray Ray that should be traced through the scene
 @return Color at the intersection point
 */

glm::vec3 trace_ray(Ray ray, int jump = 0)
{ // add "jump", to keep track of recursion
	// lecture 8 page 44

	// add if to limit recursion
	// like a while between function
	if (jump > MAX_JUMP)
	{
		return ambient_light;
	}
	// end our code
	Hit closest_hit;

	closest_hit.hit = false;
	closest_hit.distance = INFINITY;

	for (int k = 0; k < objects.size(); k++)
	{
		Hit hit = objects[k]->intersect(ray);
		if (hit.hit == true && hit.distance < closest_hit.distance)
			closest_hit = hit;
	}

	glm::vec3 color(0.0);
	if (closest_hit.hit)
	{
		// Apply Perlin noise to affect color or texture based on the intersection position
		glm::vec2 uv = glm::vec2(closest_hit.intersection); // Using xz coordinates for texture mapping
		float noise = perlin_noise(uv * 0.1f);				// Scale the input for smooth variation

		// Modify the color of the material based on the noise value
		glm::vec3 modified_color = closest_hit.object->color * (1.0f + 0.5f * noise);

		// Calculate Phong shading or any other material model
		color = PhongModel(closest_hit.intersection, closest_hit.normal, glm::normalize(-ray.direction), closest_hit.object->getMaterial());

		// Apply the modified color due to Perlin noise
		// color *= modified_color;

		// start our code
		glm::vec3 reflect_color(0.0);

		// there is reflectivity
		if (closest_hit.object->getMaterial().reflex > 0.0f && !closest_hit.object->getMaterial().refract_flag)
		{
			// calculate reflection direction and cast reflection ray
			glm::vec3 reflect_dir = glm::reflect(ray.direction, closest_hit.normal);
			Ray reflect_ray(closest_hit.intersection + tol * reflect_dir, reflect_dir);

			// ready to jump;
			jump++;
			reflect_color = trace_ray(reflect_ray, jump);

			// blend the phong color and reflect color based on reflectivity
			color = color * (1 - closest_hit.object->getMaterial().reflex) + reflect_color * closest_hit.object->getMaterial().reflex;
		}
		// use Snell's Law applied like seen in slides (if we have refraction)
		if (closest_hit.object->getMaterial().refract_flag)
		{
			float refractive_idx = closest_hit.object->getMaterial().reflex; // Material's refractive index

			float refract_idx = closest_hit.object->getMaterial().refract_idx;

			bool outside = glm::dot(ray.direction, closest_hit.normal) >= 0;

			// calculate beta
			float beta = outside ? refract_idx : 1.0f / refract_idx;
			glm::vec3 normal_to_reflect = outside ? -closest_hit.normal : closest_hit.normal;

			// coses
			float cos_theta1 = glm::dot(ray.direction, normal_to_reflect);
			// like abs but for float
			if (cos_theta1 < 0.0f)
				cos_theta1 = -cos_theta1;

			// see if we can refract -> slide
			float cos_theta2 = sqrt(1 - (beta * beta) * (1 - cos_theta1 * cos_theta1));

			// sin
			float sin_theta1 = sqrt(1 - cos_theta1 * cos_theta1);

			// ray is outside or inside
			float delta1 = outside ? refract_idx : 1.0f;
			float delta2 = outside ? 1.0f : refract_idx;

			glm::vec3 refract_color(0.0);

			if (beta * sin_theta1 < 1.0f)
			{ // enough for refraction
				// coefficents for refracted ray. prof... why these name ::cry
				glm::vec3 a = normal_to_reflect * glm::dot(ray.direction, normal_to_reflect); // normal component
				glm::vec3 b = ray.direction - a;											  // tangential component(parallel to the surface)

				// lesson 8 page 31, refracted direction, alpha formula
				float alpha = sqrt(1 + (1 - beta * beta) * (glm::dot(b, b) / glm::dot(a, a)));

				// lesson 8, page 31, ray direction, r formula, tell prof to choose better words... a*a=alpha*a :/
				glm::vec3 rafract_dir = glm::normalize(alpha * a + beta * b);
				glm::vec3 shift_pos = closest_hit.intersection + tol * rafract_dir;
				Ray refracted_ray = Ray(shift_pos, rafract_dir);

				// ready to jump;
				jump++;
				refract_color = trace_ray(refracted_ray, jump);
			}
			// reflective part
			glm::vec3 reflect_dir = glm::reflect(ray.direction, normal_to_reflect);
			Ray new_reflect_ray(closest_hit.intersection + tol * reflect_dir, reflect_dir);

			// ready to jump;
			jump++;
			reflect_color = trace_ray(new_reflect_ray, jump);

			// fresnel effect
			float fresnel_reflect = compute_fresnel(delta1, delta2, cos_theta1, cos_theta2);
			float fresnel_refract = 1 - fresnel_reflect;

			// blend with refraction
			// NOTE color has shadow
			// see above
			// slide
			glm::vec3 I_reflect = fresnel_reflect * reflect_color;
			glm::vec3 I_refract = fresnel_refract * refract_color;
			color = I_reflect + I_refract;
		}
		// our code ends
	}
	else
	{
		color = glm::vec3(0.0, 0.0, 0.0);
	}
	return color;
}

/**
 Function defining the scene
 */
void sceneDefinition()
{
	// Materiale verde (diffuso)
	Material green_diffuse;
	green_diffuse.ambient = glm::vec3(0.2f, 0.2f, 0.2f); // Ambient light should be more subtle for realism
	green_diffuse.diffuse = glm::vec3(0.3f, 0.7f, 0.3f); // More balanced green
	green_diffuse.sigmaT = 0.05f;						 // Lower absorption to maintain color brightness
	green_diffuse.sigmaB = 0.05f;

	// Materiale rosso (speculare)
	Material red_specular;
	red_specular.ambient = glm::vec3(0.2f, 0.1f, 0.1f); // Lower ambient for more natural lighting
	red_specular.diffuse = glm::vec3(0.6f, 0.1f, 0.1f); // Stronger red diffuse color
	red_specular.specular = glm::vec3(0.5f);
	red_specular.shininess = 15.0f; // Slightly higher shininess for more glossy appearance
	red_specular.sigmaT = 0.1f;
	red_specular.sigmaB = 0.1f;

	// Materiale blu (speculare, cromatico)
	Material blue_specular;
	blue_specular.ambient = glm::vec3(0.2f, 0.2f, 0.6f); // More subtle blue ambient
	blue_specular.diffuse = glm::vec3(0.4f, 0.4f, 1.0f); // Stronger blue diffuse
	blue_specular.specular = glm::vec3(0.7f);			 // Higher specular to make it shinier
	blue_specular.shininess = 35.0f;					 // Moderate shininess
	blue_specular.reflex = 0.5f;						 // Strong reflection property
	blue_specular.sigmaT = 0.03f;
	blue_specular.sigmaB = 0.03f;

	// Materiale plastica (con rifrazione)
	Material plastic;
	plastic.ambient = glm::vec3(0.02f, 0.02f, 0.1f); // Darker ambient for plastic
	plastic.diffuse = glm::vec3(0.2f, 0.2f, 1.0f);	 // Blue diffuse
	plastic.specular = glm::vec3(0.6f);				 // Good specular reflection
	plastic.refract_flag = true;
	plastic.refract_idx = 1.5f; // Higher refractive index for plastic
	plastic.shininess = 10.0f;	// Moderate shininess for plastic
	plastic.sigmaT = 0.05f;		// Reduced absorption for realistic plastic
	plastic.sigmaB = 0.05f;

	// Materiale giallo (ceramica, meno riflettente)
	Material yellow_specular;
	yellow_specular.ambient = glm::vec3(0.1f, 0.1f, 0.0f); // Subtle yellow ambient
	yellow_specular.diffuse = glm::vec3(0.6f, 0.6f, 0.0f); // Stronger yellow diffuse
	yellow_specular.specular = glm::vec3(0.2f);			   // Lower specular reflection for ceramics
	yellow_specular.shininess = 20.0f;					   // Moderate shininess for ceramic
	yellow_specular.sigmaT = 0.1f;
	yellow_specular.sigmaB = 0.1f;

	// Oggetti della scena
	objects.push_back(new Sphere(3.0, glm::vec3(-10, 10, 20), red_specular)); // Blu speculare
	objects.push_back(new Sphere(0.5, glm::vec3(1, -1, 2), red_specular));	  // Rosso speculare
	// objects.push_back(new Sphere(2, glm::vec3(-3, -1, 8), plastic));		// Plastica (con rifrazione)

	// Piani della scena
	objects.push_back(new Plane(glm::vec3(0, -3, 0), glm::vec3(0.0, -1, 0)));
	objects.push_back(new Plane(glm::vec3(0, 1, 30), glm::vec3(0.0, 0.0, -1.0), green_diffuse)); // Verde diffuso
	objects.push_back(new Plane(glm::vec3(-15, 1, 0), glm::vec3(1.0, 0.0, 0.0), red_specular));	 // Rosso speculare
	objects.push_back(new Plane(glm::vec3(15, 1, 0), glm::vec3(-1.0, 0.0, 0.0), blue_specular)); // Blu speculare
	objects.push_back(new Plane(glm::vec3(0, 27, 0), glm::vec3(0.0, -1, 0), green_diffuse));
	objects.push_back(new Plane(glm::vec3(0, 1, -0.01), glm::vec3(0.0, 0.0, 1.0), blue_specular));

	// Luci della scena
	lights.push_back(new Light(glm::vec3(10, 15, 10), glm::vec3(1.0f)));
	lights.push_back(new Light(glm::vec3(-10, 15, 10), glm::vec3(1.0f)));
	lights.push_back(new Light(glm::vec3(-5, 5, 0), glm::vec3(0.1f)));

	Mesh *car = new Mesh("../../../Rendering Competition/code/meshes/car.obj");
	glm::mat4 translation = glm::translate(glm::vec3(0.0f, -2.0f, 10.0f));
	glm::mat4 scaling = glm::scale(glm::vec3(0.1f, 0.1f, 0.1f));
	glm::mat4 rotation = glm::rotate(glm::radians(-160.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	car->setTransformation(translation * rotation * scaling);
	objects.push_back(car);
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

int main(int argc, const char *argv[])
{

	clock_t t = clock(); // variable for keeping the time of the rendering
	clock_t start_time = clock();

	int width = 1024; // 320;  // width of the image
	int height = 768; // height of the image
	float fov = 90;	  // field of view

	sceneDefinition(); // Let's define a scene

	Image image(width, height); // Create an image where we will store the result
	vector<glm::vec3> image_values(width * height);

	float s = 2 * tan(0.5 * fov / 180 * M_PI) / width;
	float X = -s * width / 2;
	float Y = s * height / 2;

	// Random number generator for jitter
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
	// Parametri Depth of Field
	int samples_per_pixel = 16;	  // Number of samples per pixel
	float aperture_radius = 0.2f; // Apertura della lente
	float focus_distance = 6.0f;  // Distanza dal piano focale

	// Generatore di numeri casuali per l'apertura
	std::uniform_real_distribution<float> aperture_dis(-0.5f, 0.5f);

	int totalPixels = width * height;
	int iteration = 0;
#pragma omp parallel for
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			glm::vec3 pixel_color(0.0f); // Accumulator for the pixel color

			for (int sample = 0; sample < samples_per_pixel; sample++)
			{
				// Add random jitter within the pixel
				float dx = X + (i + 0.5f + dis(gen)) * s;
				float dy = Y - (j + 0.5f + dis(gen)) * s;
				float dz = 1;

				glm::vec3 focal_point = glm::vec3(dx, dy, dz) * focus_distance; // Punto sul piano focale

				// Calcolo di un punto casuale sull'apertura
				float lens_u, lens_v;
				do
				{
					lens_u = aperture_dis(gen);
					lens_v = aperture_dis(gen);
				} while (lens_u * lens_u + lens_v * lens_v > 1.0f); // Assicuriamoci che sia nel cerchio unitario

				lens_u *= aperture_radius;
				lens_v *= aperture_radius;

				glm::vec3 origin(lens_u, lens_v, 0);						// Origine del raggio spostata
				glm::vec3 direction = glm::normalize(focal_point - origin); // Direzione verso il piano focale

				Ray ray(origin, direction);
				pixel_color += trace_ray(ray); // Accumulate the color from the ray
			}

			// Average the accumulated color
			pixel_color /= static_cast<float>(samples_per_pixel);

			// Apply tone mapping and set the pixel
			image.setPixel(i, j, toneMapping(pixel_color));

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
	}
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
