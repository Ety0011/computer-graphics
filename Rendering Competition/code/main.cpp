/**
@file main.cpp
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>
#include "glm/gtx/norm.hpp"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

#include <omp.h>
#include <iomanip>
#include <algorithm>
#include <random>
#include <array>

#include "Image.h"
#include "Material.h"

#include "glm/gtc/random.hpp"

using namespace std;

// Tools

float getDensity(const glm::vec3 &position)
{
	return glm::smoothstep(0.0f, 1.0f, glm::length(position) - 5.0f);
}

glm::vec3 randomHemisphereDirection(const glm::vec3 &normal = glm::vec3(0.0f, 0.0f, 1.0f))
{
	// Use glm to generate random spherical coordinates
	float theta = glm::linearRand(0.0f, glm::two_pi<float>()); // Azimuthal angle
	float phi = glm::linearRand(0.0f, glm::half_pi<float>());  // Polar angle (only half-sphere)

	// Convert spherical coordinates to Cartesian coordinates
	glm::vec3 randomDir(
		std::sin(phi) * std::cos(theta),
		std::sin(phi) * std::sin(theta),
		std::cos(phi));

	// If a normal is provided, align the random direction with the hemisphere around the normal
	if (normal != glm::vec3(0.0f, 0.0f, 1.0f))
	{
		randomDir = glm::dot(randomDir, normal) > 0.0f ? randomDir : -randomDir;
	}

	return randomDir;
}

/**
 * Class representing a single photon.
 */
class Photon
{
public:
	glm::vec3 position;	 ///< Position of the photon
	glm::vec3 direction; ///< Direction of the photon's travel
	glm::vec3 power;	 ///< Power (color/intensity) carried by the photon

	/**
	 * Constructor for the photon.
	 * @param position Position of the photon
	 * @param direction Direction of the photon
	 * @param power Power carried by the photon
	 */
	Photon(glm::vec3 position, glm::vec3 direction, glm::vec3 power)
		: position(position), direction(direction), power(power)
	{
	}
};

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
	Ray(glm::vec3 origin, glm::vec3 direction) : origin(origin), direction(direction) {}
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

// geoemtry
// solid

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

// abstract
// Axis Aligned Bounding box
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

/*
 models effect
 */

// rays
glm::vec3 trace_ray(Ray ray);
glm::vec3 trace_ray_recursive(Ray ray, int depth_recursion);

// lights
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

// shadow
float compute_soft_shadow(const glm::vec3 &intersection_point, const glm::vec3 &random_point)
{
	// Create a shadow ray from the intersection point to the random point on the light source
	glm::vec3 light_direction = glm::normalize(random_point - intersection_point);
	Ray shadow_ray(intersection_point + 1e-4f * light_direction, light_direction); // Avoid self-intersection

	// Check for intersection with scene objects
	Hit closest_hit;
	closest_hit.hit = false;
	closest_hit.distance = INFINITY;

	// 3. Check if light_direction is within the cone
	float cos_cone_angle = glm::cos(glm::radians(80.0f));
	glm::vec3 cone_direction = glm::vec3(0.0f, -1.0f, 0.0f); // Downward cone
	if (glm::dot(-light_direction, cone_direction) < cos_cone_angle && glm::dot(-light_direction, cone_direction) > -0.01)
	{
		return 0.0f; // Skip rays outside the cone
	}

	for (int k = 0; k < objects.size(); k++)
	{
		Hit hit = objects[k]->intersect(shadow_ray);
		if (hit.hit && hit.distance < closest_hit.distance)
		{
			closest_hit = hit;
		}
	}

	// If no object blocks the light, the ray is unblocked
	return (closest_hit.hit && closest_hit.distance < glm::distance(intersection_point, random_point)) ? 0.0f : 1.0f;
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
	int light_samples = 16;

	for (int light_num = 0; light_num < lights.size(); light_num++)
	{
		glm::vec3 light_position = lights[light_num]->position;
		float light_radius = lights[light_num]->radius;
		glm::vec3 color_contribution(0.0f); // To accumulate the color contribution from the light

		// Sample multiple points on the light's square surface
		for (int sample_num = 0; sample_num < light_samples; sample_num++)
		{
			// Sample a random point on the square light source
			glm::vec3 random_point = light_position + random_point_on_square(light_radius);

			// Calculate direction to the light sample point
			glm::vec3 to_light_dir = glm::normalize(random_point - point);
			glm::vec3 reflected_from_light_dir = glm::reflect(-to_light_dir, normal);
			float light_distance = glm::distance(point, random_point);

			// Compute diffuse and specular components
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

			// Compute the soft shadow visibility term for this point on the light
			float visibility = compute_soft_shadow(point, random_point);

			// Attenuation based on distance
			float r = max(light_distance, 0.1f);
			glm::vec3 light_contribution = lights[light_num]->color * (diffuse + specular) * visibility / pow(r, 2.0f);

			// Accumulate the light contribution from this sample point
			color_contribution += light_contribution;
		}

		// Average the contributions from all samples for this light
		color += color_contribution / float(light_samples);
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
 * Class representing a photon map.
 */
class PhotonMap
{
private:
	std::vector<Photon> photons; ///< Collection of photons in the map

public:
	/**
	 * Add a photon to the map.
	 * @param photon The photon to add
	 */
	void addPhoton(const Photon &photon)
	{
		photons.push_back(photon);
	}

	/**
	 * Estimate radiance at a given point using nearby photons.
	 * @param position The position to estimate radiance
	 * @param normal The surface normal at the position
	 * @param radius The radius within which to consider photons
	 * @return The estimated radiance
	 */
	glm::vec3 estimateRadiance(glm::vec3 &position, glm::vec3 &normal, float radius)
	{
		glm::vec3 radiance(0.0f);
		float radiusSquared = radius * radius;

		for (Photon &photon : photons)
		{
			float distanceSquared = glm::distance2(position, photon.position);
			if (distanceSquared <= radiusSquared && glm::dot(normal, photon.direction) < 0)
			{
				radiance += photon.power / static_cast<float>(M_PI * radiusSquared);
			}
		}

		return radiance;
	}
};

/**
 * Function to trace photons into the scene.
 * @param scene The scene containing objects
 * @param photonMap The photon map to store traced photons
 * @param light The light source emitting photons
 * @param numPhotons Number of photons to trace
 */
void tracePhotons(const std::vector<Object *> &scene, PhotonMap &photonMap, const Light &light, int numPhotons)
{
	for (int i = 0; i < numPhotons; i++)
	{
		// Generate a random direction for photon emission
		glm::vec3 direction = randomHemisphereDirection(light.position);
		glm::vec3 power = light.color / static_cast<float>(numPhotons);

		Ray photonRay(light.position, glm::normalize(direction));

		for (Object *object : scene)
		{
			Hit hit = object->intersect(photonRay);
			if (hit.hit)
			{
				// Store the photon at the hit point
				photonMap.addPhoton(Photon(hit.intersection, photonRay.direction, power));
				// Scatter photon (diffuse or specular reflection)
				photonRay.origin = hit.intersection;
				photonRay.direction = randomHemisphereDirection(hit.normal);
				power *= object->getMaterial().diffuse; // Adjust power based on material properties
			}
		}
	}
}

/**
 * Integrate photon mapping into rendering.
 * @param ray The ray to trace
 * @param photonMap The photon map
 * @param maxDepth Maximum recursion depth
 * @return The computed color for the ray
 */
glm::vec3 renderWithPhotonMapping(Ray &ray, PhotonMap &photonMap, const std::vector<Object *> &scene, int maxDepth, float radius)
{
	Hit hit;
	glm::vec3 color(0.0f);
	int depth = 0;

	// Traverse the scene for intersections
	while (depth < maxDepth)
	{
		bool foundHit = false;
		for (Object *object : scene)
		{
			hit = object->intersect(ray);
			if (hit.hit)
			{
				foundHit = true;

				// Direct illumination
				color += object->getMaterial().diffuse * photonMap.estimateRadiance(hit.intersection, hit.normal, radius);

				// Generate reflection or transmission ray
				Ray nextRay(hit.intersection, glm::reflect(ray.direction, hit.normal));
				color += renderWithPhotonMapping(nextRay, photonMap, scene, maxDepth - 1, radius);

				break;
			}
		}

		if (!foundHit) // No intersection found
		{
			break;
		}

		depth++;
	}

	return color; // Return accumulated color
}

// Generate smoke particles in a hemisphere
std::vector<std::array<double, 3>> generateSmoke(size_t numParticles)
{
	std::vector<std::array<double, 3>> smokeParticles;

	for (size_t i = 0; i < numParticles; ++i)
	{
		auto dir = randomHemisphereDirection();
		double scale = 0.1;
		smokeParticles.push_back({scale * dir[0], scale * dir[1], scale * dir[2]});
	}

	return smokeParticles;
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
	float shininess_plane = 0.2f;
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

	Material yellow_plane;
	green_plane.diffuse = glm::vec3(0.8f, 0.8f, 0.3f);
	green_plane.ambient = green_plane.diffuse / glm::vec3(10);
	green_plane.specular = glm::vec3(0.0);
	green_plane.shininess = shininess_plane;

	Material green_sphere;
	green_sphere.diffuse = glm::vec3(0.2f, 1.0f, 0.2f);
	green_sphere.ambient = green_sphere.diffuse / glm::vec3(10);
	green_sphere.specular = glm::vec3(0.5);
	green_sphere.shininess = shininess_sphere;
	green_sphere.is_anisotropic = true;
	green_sphere.alpha_x = 0.8f;

	// Define smoke material (for visualization in the scene)
	Material smoke_material;
	smoke_material.diffuse = glm::vec3(0.1f, 0.1f, 0.1f); // Light gray color for smoke
	smoke_material.ambient = smoke_material.diffuse / glm::vec3(1);
	smoke_material.specular = glm::vec3(0.0f);
	smoke_material.shininess = 0.0f;

	// Add the smoke cube as a bounding volume for the smoke effect (no new object, just volume definition)
	glm::vec3 smoke_min(-0.1f, -0.1f, 0.3f);
	glm::vec3 smoke_max(0.05f, 0.05f, 0.07f);

	// Semi-transparent material with Fresnel effect
	Material transparent_material;
	transparent_material.diffuse = glm::vec3(0.3f, 0.3f, 1.0f); // Semi-transparent blue color
	transparent_material.ambient = transparent_material.diffuse / glm::vec3(5);
	transparent_material.specular = glm::vec3(0.1f);
	transparent_material.shininess = 0.2f;	// Semi-transparent
	transparent_material.reflection = 0.2f; // Fresnel effect based on refraction index

	Material reflex;
	reflex.diffuse = glm::vec3(0.3f, 0.3f, 1.0f); // Semi-transparent blue color
	reflex.ambient = transparent_material.diffuse / glm::vec3(5);
	reflex.specular = glm::vec3(0.5f);
	reflex.shininess = 0.2f;  // Semi-transparent
	reflex.reflection = 0.2f; // Fresnel effect based on refraction index

	// Spheres
	objects.push_back(new Sphere(1.0, glm::vec3(-2, -1, 5), green_sphere));
	objects.push_back(new Sphere(0.7, glm::vec3(0, -2.5, 4), smoke_material));
	objects.push_back(new Sphere(0.5, glm::vec3(-1.5, -2.5, 3), red_sphere));
	objects.push_back(new Sphere(0.5, glm::vec3(-1.5, 1.6, 2.3), reflex));

	// Lights
	lights.push_back(new Light(glm::vec3(0, 2.99, 4), glm::vec3(0.05)));

	// Planes
	// planes above and below
	objects.push_back(new Plane(glm::vec3(0, -3, 0), glm::vec3(0.0, 1, 0), blue_plane));
	objects.push_back(new Plane(glm::vec3(0, 3, 0), glm::vec3(0.0, -1, 0), green_plane));

	// planes right and left
	objects.push_back(new Plane(glm::vec3(-3, 0, 0), glm::vec3(1.0, 0.0, 0.0), transparent_material));
	objects.push_back(new Plane(glm::vec3(3, 0, 0), glm::vec3(-1.0, 0.0, 0.0), red_plane));

	// plane front
	objects.push_back(new Plane(glm::vec3(0, 0, 6), glm::vec3(0.0, 0.0, -1.0), yellow_plane));

	glm::mat4 translation;
	glm::mat4 scaling;
	glm::mat4 rotation;

	Mesh *cube = new Mesh("./meshes/cube.obj");
	translation = glm::translate(glm::vec3(1.5f, -2.5f, 3.5f));
	scaling = glm::scale(glm::vec3(0.9f));
	rotation = glm::rotate(glm::radians(20.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	cube->setTransformation(translation * rotation * scaling);
	objects.push_back(cube);

	Cone *cone = new Cone(transparent_material);
	translation = glm::translate(glm::vec3(1.0f, -1.5f, 4.5f));
	scaling = glm::scale(glm::vec3(0.5f, 3.0f, 0.5f));
	rotation = glm::rotate(glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	cone->setTransformation(translation * rotation * scaling);
	cone->setMaterial(green_plane);
	objects.push_back(cone);

	Mesh *ring = new Mesh("./meshes/ring.obj");
	translation = glm::translate(glm::vec3(1.0f, 1.0f, 4.5f));
	scaling = glm::scale(glm::vec3(0.7f));
	rotation = glm::rotate(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	ring->setTransformation(translation * rotation * scaling);
	ring->setMaterial(yellow_plane);
	objects.push_back(ring);

	Mesh *iso = new Mesh("./meshes/isocat.obj");
	translation = glm::translate(glm::vec3(-1.0f, 1.7f, 4.5f));
	scaling = glm::scale(glm::vec3(0.5f));
	rotation = glm::rotate(glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	iso->setMaterial(transparent_material);
	iso->setTransformation(translation * rotation * scaling);
	objects.push_back(iso);
}

glm::vec3 toneMapping(glm::vec3 intensity)
{
	float gamma = 1.0 / 2.0;
	float alpha = 12.0f;
	return glm::clamp(alpha * glm::pow(intensity, glm::vec3(gamma)), glm::vec3(0.0), glm::vec3(1.0));
}

// smoke

glm::vec3 renderSmoke(const glm::vec3 &rayOrigin, const glm::vec3 &rayDirection, const float stepSize)
{
	glm::vec3 color(0.0f);
	glm::vec3 attenuation(1.0f); // Track light attenuation

	for (float t = 0.0f; t < 1.0f; t += stepSize)
	{
		glm::vec3 position = rayOrigin + t * rayDirection;

		// Compute density at this position (in a real implementation, interpolate from the grid)
		float density = getDensity(position);

		// Add contribution to the color
		color += attenuation * density;

		// Attenuate light
		attenuation *= glm::vec3(1.0f - density * stepSize);
	}

	return color;
}

// deph of field

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
	omp_set_num_threads(12);

	clock_t t = clock(); // variable for keeping the time of the rendering
	clock_t start_time = clock();

	int width = 100;  // width of the image
	int height = 100; // height of the image
	float fov = 90;	  // field of view

	sceneDefinition(); // Let's define a scene

	Image image(width, height); // Create an image where we will store the result
	vector<glm::vec3> image_values(width * height);

	float s = 2 * tan(0.5 * fov / 180 * M_PI) / width;
	float X = -s * width / 2;
	float Y = s * height / 2;

	int totalPixels = width * height;
	int iteration = 0;

	int aa_samples = 4;	 // Supersampling for anti-aliasing
	int dof_samples = 3; // Number of samples for depth of field
	// DEFAULT 0.5f
	float aperture_radius = 0.05f; // Controls the size of the blur (lens aperture)
	// DEFAULT 8.0f
	float focal_length = 3.0f; // Distance to the focal plane
	size_t numParticles = 3;
	
	// Generate smoke
	auto smoke = generateSmoke(numParticles);

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
