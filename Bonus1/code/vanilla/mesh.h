#pragma once

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <array>

#include "../glm/glm.hpp"
#include "../glm/gtx/transform.hpp"
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

  void setTransformationTriangles(glm::mat4 matrix)
  {
    for (auto &triangle : triangles)
    {
      triangle.setTransformation(matrix);
    }
  }

  bool loadFromFile(const std::string &filename)
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      std::cerr << "Failed to open file: " << filename << std::endl;
      return false;
    }

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
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

        std::array<glm::vec3, 3> triangleVertices;
        for (int i = 0; i < 3; i++)
        {
          triangleVertices[i] = vertices[verticesIndex[i]];
        }

        Triangle triangle(triangleVertices, this->material);
        if (smoothShading)
        {
          std::array<glm::vec3, 3> triangleNormals;
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

  Hit intersect(Ray ray)
  {
    Hit closest_hit;
    closest_hit.hit = false;
    closest_hit.distance = INFINITY;

    for (auto triangle : triangles)
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
      closest_hit.intersection = transformationMatrix * glm::vec4(closest_hit.intersection, 1.0);
      closest_hit.normal = glm::normalize(normalMatrix * glm::vec4(closest_hit.normal, 0.0));
      closest_hit.distance = glm::length(closest_hit.intersection - ray.origin);
    }

    return closest_hit;
  }
};

std::vector<Object *> defineObjects() {
  std::vector<Object *> objects; 

	Material green_diffuse;
	green_diffuse.ambient = glm::vec3(0.7f, 0.9f, 0.7f);
	green_diffuse.diffuse = glm::vec3(0.7f, 0.9f, 0.7f);

	Material red_specular;
	red_specular.ambient = glm::vec3(1.0f, 0.3f, 0.3f);
	red_specular.diffuse = glm::vec3(1.0f, 0.3f, 0.3f);
	red_specular.specular = glm::vec3(0.5);
	red_specular.shininess = 10.0;

	Material blue_specular;
	blue_specular.ambient = glm::vec3(0.7f, 0.7f, 1.0f);
	blue_specular.diffuse = glm::vec3(0.7f, 0.7f, 1.0f);
	blue_specular.specular = glm::vec3(0.6);
	blue_specular.shininess = 100.0;

	// Material green_diffuse;
	green_diffuse.ambient = glm::vec3(0.03f, 0.1f, 0.03f);
	green_diffuse.diffuse = glm::vec3(0.3f, 1.0f, 0.3f);

	// Material red_specular;
	red_specular.diffuse = glm::vec3(1.0f, 0.2f, 0.2f);
	red_specular.ambient = glm::vec3(0.01f, 0.02f, 0.02f);
	red_specular.specular = glm::vec3(0.5);
	red_specular.shininess = 10.0;

	// Material blue_specular;
	blue_specular.ambient = glm::vec3(0.02f, 0.02f, 0.1f);
	blue_specular.diffuse = glm::vec3(0.2f, 0.2f, 1.0f);
	blue_specular.specular = glm::vec3(0.6);
	blue_specular.shininess = 100.0;

	Material red_diffuse;
	red_diffuse.ambient = glm::vec3(0.09f, 0.06f, 0.06f);
	red_diffuse.diffuse = glm::vec3(0.9f, 0.6f, 0.6f);

	Material blue_diffuse;
	blue_diffuse.ambient = glm::vec3(0.06f, 0.06f, 0.09f);
	blue_diffuse.diffuse = glm::vec3(0.6f, 0.6f, 0.9f);

	// Define material for the mesh
	Material mesh_material;
	mesh_material.ambient = glm::vec3(0.1f, 0.1f, 0.1f);
	mesh_material.diffuse = glm::vec3(0.6f, 0.7f, 0.8f);
	mesh_material.specular = glm::vec3(0.9f);
	mesh_material.shininess = 50.0f;

	// Mesh
	Mesh *mesh = new Mesh("/home/leonardo/meshes/bunny_small.obj", mesh_material);
	glm::mat4 translation = glm::translate(glm::vec3(0.0f, -1.0f, 5.0f));
	glm::mat4 scaling = glm::scale(glm::vec3(1.0f, 1.0f, 1.0f));
	mesh->setTransformation(translation * scaling);
	mesh->setTransformationTriangles(translation * scaling);

	objects.push_back(new Plane(glm::vec3(0, -3, 0), glm::vec3(0.0, 1, 0)));
	objects.push_back(new Plane(glm::vec3(0, 1, 30), glm::vec3(0.0, 0.0, -1.0), green_diffuse));
	objects.push_back(new Plane(glm::vec3(-15, 1, 0), glm::vec3(1.0, 0.0, 0.0), red_diffuse));
	objects.push_back(new Plane(glm::vec3(15, 1, 0), glm::vec3(-1.0, 0.0, 0.0), blue_diffuse));
	objects.push_back(new Plane(glm::vec3(0, 27, 0), glm::vec3(0.0, -1, 0)));
	objects.push_back(new Plane(glm::vec3(0, 1, -0.01), glm::vec3(0.0, 0.0, 1.0), green_diffuse));

	objects.push_back(mesh);

  return objects;
}
