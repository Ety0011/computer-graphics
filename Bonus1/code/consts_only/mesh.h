#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <array>

#include "object.h"

class Mesh : public Object
{
private:
  std::vector<Triangle> triangles;
  bool smoothShading;

public:
  Mesh(Material material)
  {
    this->material = material;
    Triangle triangle(material);
    triangles.push_back(triangle);
  }
  Mesh(const std::string &filename, Material material)
  {
    if (!loadFromFile(filename))
      exit(-1);

    this->material = material;
  }

  void setTransformationTriangles(raym::mat4 matrix)
  {
    for (Triangle &triangle : triangles)
      triangle.setTransformation(matrix);
  }

  // ! Per ora la lascio cosi', anche se potrebbe essere lenta per mesh grandi
  bool loadFromFile(const std::string &filename)
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      std::cerr << "Failed to open file: " << filename << std::endl;
      return false;
    }

    std::vector<raym::vec3> vertices;
    std::vector<raym::vec3> normals;
    std::string line;
    while (getline(file, line))
    {
      std::istringstream lineStream(line);
      std::string prefix;
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
        std::vector<int> verticesIndex;
        std::vector<int> normalsIndex;
        std::string indexPair;

        while (lineStream >> indexPair)
        {
          std::istringstream indexStream(indexPair);
          int vIndex, vnIndex;

          indexStream >> vIndex;
          verticesIndex.push_back(vIndex - 1);

          if (indexStream.peek() == '/')
          {
            indexStream.ignore(2);
            indexStream >> vnIndex;
            normalsIndex.push_back(vnIndex - 1);
          }
        }

        std::array<raym::vec3, 3> triangleVertices;
        for (int i = 0; i < 3; i++)
        {
          triangleVertices[i] = vertices[verticesIndex[i]];
        }

        Triangle triangle(triangleVertices, this->material);
        if (smoothShading)
        {
          std::array<raym::vec3, 3> triangleNormals;
          for (int i = 0; i < 3; i++)
          {
            triangleNormals[i] = normals[normalsIndex[i]];
          }
          triangle = Triangle(triangleVertices, triangleNormals, this->material);
        }

        triangles.push_back(triangle);
      }
    }
    file.close();
    return true;
  }

  constexpr raym::vec3 _closestIntersection(const Hit& hit) {
    return raym::to_vec3(transformationMatrix * raym::vec4(hit.intersection, 1.0f));
  }

  constexpr raym::vec3 _closestNormal(const Hit& hit) {
    return raym::to_vec3(raym::normalize(normalMatrix * raym::vec4(hit.normal, 0.0f)));
  }

  constexpr float _closestDistance(const Hit& hit, const Ray& ray) {
    return raym::length(hit.intersection - ray.origin);
  }

  Hit intersect(const Ray& ray)
  {
    Hit closest_hit;
    closest_hit.hit = false;
    closest_hit.distance = INFINITY;

    for (Triangle triangle : triangles)
    {
      Hit hit = triangle.intersect(ray);
      if (hit.hit && hit.distance < closest_hit.distance)
      {
        closest_hit = hit;
        closest_hit.object = this;
      }
    }

    if (closest_hit.hit)
    {
      closest_hit.intersection = _closestIntersection(closest_hit);
      closest_hit.normal = _closestNormal(closest_hit);
      closest_hit.distance = _closestDistance(closest_hit, ray);
    }

    return closest_hit;
  }
};

inline std::vector<Object *> defineObjects() {
  std::vector<Object *> objects; 

	Material green_diffuse;
	green_diffuse.ambient = raym::vec3(0.7f, 0.9f, 0.7f);
	green_diffuse.diffuse = raym::vec3(0.7f, 0.9f, 0.7f);

	Material red_specular;
	red_specular.ambient = raym::vec3(1.0f, 0.3f, 0.3f);
	red_specular.diffuse = raym::vec3(1.0f, 0.3f, 0.3f);
	red_specular.specular = raym::vec3(0.5);
	red_specular.shininess = 10.0;

	Material blue_specular;
	blue_specular.ambient = raym::vec3(0.7f, 0.7f, 1.0f);
	blue_specular.diffuse = raym::vec3(0.7f, 0.7f, 1.0f);
	blue_specular.specular = raym::vec3(0.6);
	blue_specular.shininess = 100.0;

	// Material green_diffuse;
	green_diffuse.ambient = raym::vec3(0.03f, 0.1f, 0.03f);
	green_diffuse.diffuse = raym::vec3(0.3f, 1.0f, 0.3f);

	// Material red_specular;
	red_specular.diffuse = raym::vec3(1.0f, 0.2f, 0.2f);
	red_specular.ambient = raym::vec3(0.01f, 0.02f, 0.02f);
	red_specular.specular = raym::vec3(0.5);
	red_specular.shininess = 10.0;

	// Material blue_specular;
	blue_specular.ambient = raym::vec3(0.02f, 0.02f, 0.1f);
	blue_specular.diffuse = raym::vec3(0.2f, 0.2f, 1.0f);
	blue_specular.specular = raym::vec3(0.6);
	blue_specular.shininess = 100.0;

	Material red_diffuse;
	red_diffuse.ambient = raym::vec3(0.09f, 0.06f, 0.06f);
	red_diffuse.diffuse = raym::vec3(0.9f, 0.6f, 0.6f);

	Material blue_diffuse;
	blue_diffuse.ambient = raym::vec3(0.06f, 0.06f, 0.09f);
	blue_diffuse.diffuse = raym::vec3(0.6f, 0.6f, 0.9f);

	// Define material for the mesh
	Material mesh_material;
	mesh_material.ambient = raym::vec3(0.1f, 0.1f, 0.1f);
	mesh_material.diffuse = raym::vec3(0.6f, 0.7f, 0.8f);
	mesh_material.specular = raym::vec3(0.9f);
	mesh_material.shininess = 50.0f;

	// Mesh
	Mesh *mesh = new Mesh("/home/leonardo/meshes/bunny_small.obj", mesh_material);
	raym::mat4 translation = raym::translate(raym::vec3(0.0f, -1.0f, 5.0f));
	raym::mat4 scaling = raym::scale(raym::vec3(1.0f, 1.0f, 1.0f));
	mesh->setTransformation(translation * scaling);
	mesh->setTransformationTriangles(translation * scaling);

	objects.push_back(new Plane(raym::vec3(0, -3, 0), raym::vec3(0.0, 1, 0)));
	objects.push_back(new Plane(raym::vec3(0, 1, 30), raym::vec3(0.0, 0.0, -1.0), green_diffuse));
	objects.push_back(new Plane(raym::vec3(-15, 1, 0), raym::vec3(1.0, 0.0, 0.0), red_diffuse));
	objects.push_back(new Plane(raym::vec3(15, 1, 0), raym::vec3(-1.0, 0.0, 0.0), blue_diffuse));
	objects.push_back(new Plane(raym::vec3(0, 27, 0), raym::vec3(0.0, -1, 0)));
	objects.push_back(new Plane(raym::vec3(0, 1, -0.01), raym::vec3(0.0, 0.0, 1.0), green_diffuse));

	objects.push_back(mesh);

  return objects;
}
