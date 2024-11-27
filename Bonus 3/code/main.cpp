/**
@file main.cpp
*/

#include <array>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>
#include "glm/geometric.hpp"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"

#include "Image.h"
#include "Material.h"
#include <algorithm>

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
		float txmin = (minBounds.x - ray.origin.x) / ray.direction.x;
		float txmax = (maxBounds.x - ray.origin.x) / ray.direction.x;
		if (txmin > txmax)
			std::swap(txmin, txmax);

		float tymin = (minBounds.y - ray.origin.y) / ray.direction.y;
		float tymax = (maxBounds.y - ray.origin.y) / ray.direction.y;
		if (tymin > tymax)
			std::swap(tymin, tymax);

		float tzmin = (minBounds.z - ray.origin.z) / ray.direction.z;
		float tzmax = (maxBounds.z - ray.origin.z) / ray.direction.z;
		if (tzmin > tzmax)
			std::swap(tzmin, tzmax);

		int overlapCount = 0;

		if (txmin <= tymax && txmax >= tymin)
		{
			overlapCount++;
		}

		if (txmin <= tzmax && txmax >= tzmin)
		{
			overlapCount++;
		}

		if (tymin <= tzmax && tymax >= tzmin)
		{
			overlapCount++;
		}

		return overlapCount >= 2;
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
			throw std::runtime_error("Failed to load mesh from file: " + filename);
		}
		bvhRoot = buildBVH(triangles);
	}
	Mesh(const string &filename, Material material)
	{
		this->material = material;
		smoothShading = 0;
		if (!loadFromFile(filename))
		{
			throw std::runtime_error("Failed to load mesh from file: " + filename);
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

		std::sort(
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
		glm::vec3 reflected_direction = glm::reflect(-light_direction, normal);

		float NdotL = glm::clamp(glm::dot(normal, light_direction), 0.0f, 1.0f);
		float VdotR = glm::clamp(glm::dot(view_direction, reflected_direction), 0.0f, 1.0f);

		glm::vec3 diffuse_color = material.diffuse;
		glm::vec3 diffuse = diffuse_color * glm::vec3(NdotL);
		glm::vec3 specular = material.specular * glm::vec3(pow(VdotR, material.shininess));

		float r = glm::distance(point, lights[light_num]->position);
		r = max(r, 0.1f);
		color += lights[light_num]->color * (diffuse + specular) / r / r;
	}
	color += ambient_light * material.ambient;
	color = glm::clamp(color, glm::vec3(0.0), glm::vec3(1.0));
	return color;
}

/**
 Functions that computes a color along the ray
 @param ray Ray that should be traced through the scene
 @return Color at the intersection point
 */
glm::vec3 trace_ray(Ray ray)
{

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
		color = PhongModel(closest_hit.intersection, closest_hit.normal, glm::normalize(-ray.direction), closest_hit.object->getMaterial());
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
	glm::mat4 translation;
	glm::mat4 scaling;
	glm::mat4 rotation;

	Mesh *bunny = new Mesh("../../../Bonus 3/code/meshes/bunny_small.obj");
	translation = glm::translate(glm::vec3(0.0f, -3.0f, 8.0f));
	bunny->setTransformation(translation);

	Mesh *armadillo = new Mesh("../../../Bonus 3/code/meshes/armadillo_small.obj");
	translation = glm::translate(glm::vec3(-4.0f, -3.0f, 10.0f));
	armadillo->setTransformation(translation);

	Mesh *lucy = new Mesh("../../../Bonus 3/code/meshes/lucy_small.obj");
	translation = glm::translate(glm::vec3(4.0f, -3.0f, 10.0f));
	lucy->setTransformation(translation);

	lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(1.0, 1.0, 1.0)));
	lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.1)));
	lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4)));

	objects.push_back(new Plane(glm::vec3(0, -3, 0), glm::vec3(0.0, 1, 0)));
	objects.push_back(new Plane(glm::vec3(0, 1, 30), glm::vec3(0.0, 0.0, -1.0)));
	objects.push_back(new Plane(glm::vec3(-15, 1, 0), glm::vec3(1.0, 0.0, 0.0)));
	objects.push_back(new Plane(glm::vec3(15, 1, 0), glm::vec3(-1.0, 0.0, 0.0)));
	objects.push_back(new Plane(glm::vec3(0, 27, 0), glm::vec3(0.0, -1, 0)));
	objects.push_back(new Plane(glm::vec3(0, 1, -0.01), glm::vec3(0.0, 0.0, 1.0)));

	objects.push_back(bunny);
	objects.push_back(armadillo);
	objects.push_back(lucy);
}
glm::vec3 toneMapping(glm::vec3 intensity)
{
	float gamma = 1.0 / 2.0;
	float alpha = 12.0f;
	return glm::clamp(alpha * glm::pow(intensity, glm::vec3(gamma)), glm::vec3(0.0), glm::vec3(1.0));
}

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

	int width = 320;  // width of the image
	int height = 240; // height of the image
	float fov = 90;	  // field of view

	sceneDefinition(); // Let's define a scene

	Image image(width, height); // Create an image where we will store the result
	vector<glm::vec3> image_values(width * height);

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

			glm::vec3 origin(0, 0, 0);
			glm::vec3 direction(dx, dy, dz);
			direction = glm::normalize(direction);

			Ray ray(origin, direction);
			image.setPixel(i, j, toneMapping(trace_ray(ray)));

			processed++;
			if (processed % (totalPixels / 100) == 0)
				printProgress((float)processed / totalPixels);
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
