/**
@file main.cpp
*/

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

using namespace std;

/**
 Class representing a single ray.
 */
class Ray{
public:
    glm::vec3 origin; ///< Origin of the ray
    glm::vec3 direction; ///< Direction of the ray
	/**
	 Contructor of the ray
	 @param origin Origin of the ray
	 @param direction Direction of the ray
	 */
    Ray(glm::vec3 origin, glm::vec3 direction) : origin(origin), direction(direction){
    }
};


class Object;

/**
 Structure representing the even of hitting an object
 */
struct Hit{
    bool hit; ///< Boolean indicating whether there was or there was no intersection with an object
    glm::vec3 normal; ///< Normal vector of the intersected object at the intersection point
    glm::vec3 intersection; ///< Point of Intersection
    float distance; ///< Distance from the origin of the ray to the intersection point
    Object *object; ///< A pointer to the intersected object
};

/**
 General class for the object
 */
class Object{
	
protected:
	glm::mat4 transformationMatrix; ///< Matrix representing the transformation from the local to the global coordinate system
	glm::mat4 inverseTransformationMatrix; ///< Matrix representing the transformation from the global to the local coordinate system
	glm::mat4 normalMatrix; ///< Matrix for transforming normal vectors from the local to the global coordinate system
	
public:
	glm::vec3 color; ///< Color of the object
	Material material; ///< Structure describing the material of the object
	/** A function computing an intersection, which returns the structure Hit */
    virtual Hit intersect(Ray ray) = 0;

	/** Function that returns the material struct of the object*/
	Material getMaterial(){
		return material;
	}
	/** Function that set the material
	 @param material A structure describing the material of the object
	*/
	void setMaterial(Material material){
		this->material = material;
	}
	/** Functions for setting up all the transformation matrices
	@param matrix The matrix representing the transformation of the object in the global coordinates */
	void setTransformation(glm::mat4 matrix){
			
		transformationMatrix = matrix;

		inverseTransformationMatrix = glm::inverse(matrix);
		normalMatrix = glm::transpose(inverseTransformationMatrix);
	}
};

/**
 Implementation of the class Object for sphere shape.
 */
class Sphere : public Object{
private:
    float radius; ///< Radius of the sphere
    glm::vec3 center; ///< Center of the sphere

public:
	/**
	 The constructor of the sphere
	 @param radius Radius of the sphere
	 @param center Center of the sphere
	 @param color Color of the sphere
	 */
    Sphere(float radius, glm::vec3 center, glm::vec3 color) : radius(radius), center(center){
		this->color = color;
    }
	Sphere(float radius, glm::vec3 center, Material material) : radius(radius), center(center){
		this->material = material;
	}
	/** Implementation of the intersection function*/
    Hit intersect(Ray ray){

        glm::vec3 c = center - ray.origin;

        float cdotc = glm::dot(c,c);
        float cdotd = glm::dot(c, ray.direction);

        Hit hit;

        float D = 0;
		if (cdotc > cdotd*cdotd){
			D =  sqrt(cdotc - cdotd*cdotd);
		}
        if(D<=radius){
            hit.hit = true;
            float t1 = cdotd - sqrt(radius*radius - D*D);
            float t2 = cdotd + sqrt(radius*radius - D*D);

            float t = t1;
            if(t<0) t = t2;
            if(t<0){
                hit.hit = false;
                return hit;
            }

			hit.intersection = ray.origin + t * ray.direction;
			hit.normal = glm::normalize(hit.intersection - center);
			hit.distance = glm::distance(ray.origin, hit.intersection);
			hit.object = this;
        }
		else{
            hit.hit = false;
		}
		return hit;
    }
};

class Plane : public Object{

private:
	glm::vec3 normal;
	glm::vec3 point;

public:
	Plane(glm::vec3 point, glm::vec3 normal) : point(point), normal(normal){
	}
	Plane(glm::vec3 point, glm::vec3 normal, Material material) : point(point), normal(normal){
		this->material = material;
	}
	Hit intersect(Ray ray){
		
		Hit hit;
		hit.hit = false;
		
        float DdotN = glm::dot(ray.direction, normal);
        if(DdotN < 0){
            
            float PdotN = glm::dot (point-ray.origin, normal);
            float t = PdotN/DdotN;
            
            if(t > 0){
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

class Cone : public Object{
private:
	Plane *plane;
	
public:
	Cone(Material material){
		this->material = material;
		plane = new Plane(glm::vec3(0,1,0), glm::vec3(0.0,1,0));
	}
	Hit intersect(Ray ray){
		
		Hit hit;
		hit.hit = false;
		
		glm::vec3 d = inverseTransformationMatrix * glm::vec4(ray.direction, 0.0); //implicit cast to vec3
		glm::vec3 o = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0); //implicit cast to vec3
		d = glm::normalize(d);
		
		
		float a = d.x*d.x + d.z*d.z - d.y*d.y;
		float b = 2 * (d.x * o.x + d.z * o.z - d.y * o.y);
		float c = o.x * o.x + o.z * o.z - o.y * o.y;
		
		float delta = b*b - 4 * a * c;
		
		if(delta < 0){
			return hit;
		}
		
		float t1 = (-b-sqrt(delta)) / (2*a);
		float t2 = (-b+sqrt(delta)) / (2*a);
		
		float t = t1;
		hit.intersection = o + t*d;
		if(t<0 || hit.intersection.y>1 || hit.intersection.y<0){
			t = t2;
			hit.intersection = o + t*d;
			if(t<0 || hit.intersection.y>1 || hit.intersection.y<0){
				return hit;
			}
		};
	
		hit.normal = glm::vec3(hit.intersection.x, -hit.intersection.y, hit.intersection.z);
		hit.normal = glm::normalize(hit.normal);
	
		
		Ray new_ray(o,d);
		Hit hit_plane = plane->intersect(new_ray);
		if(hit_plane.hit && hit_plane.distance < t && length(hit_plane.intersection - glm::vec3(0,1,0)) <= 1.0 ){
			hit.intersection = hit_plane.intersection;
			hit.normal = hit_plane.normal;
		}
		
		hit.hit = true;
		hit.object = this;
		hit.intersection = transformationMatrix * glm::vec4(hit.intersection, 1.0); //implicit cast to vec3
		hit.normal = (normalMatrix * glm::vec4(hit.normal, 0.0)); //implicit cast to vec3
		hit.normal = glm::normalize(hit.normal);
		hit.distance = glm::length(hit.intersection - ray.origin);
		
		return hit;
	}
};

class Triangle : public Object{
private:
	Plane *plane;
	
public:
	glm::vec3 p1;
	glm::vec3 p2;
	glm::vec3 p3;

	Triangle(Material material){
		this->material = material;
		plane = new Plane(glm::vec3(0,0,0), glm::vec3(0,0,-1));
		p1 = glm::vec3(-0.5, -0.5, 0);
		p2 = glm::vec3(0.5, -0.5, 0);
		p3 = glm::vec3(0, 0.5, 0);
	}
	Hit intersect(Ray ray){
		
		Hit hit;
		hit.hit = false;
		
		glm::vec3 tOrigin = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0);
		glm::vec3 tDirection = inverseTransformationMatrix * glm::vec4(ray.direction, 0.0);
		tDirection = glm::normalize(tDirection);
		
		Hit hitPlane = plane->intersect(Ray(tOrigin, tDirection));
		if(!hitPlane.hit) {
			return hit;
		}
		hit.intersection = hitPlane.intersection;
		hit.normal = hitPlane.normal;

		glm::vec3 p = hit.intersection;
		glm::vec3 N = glm::cross(p2 - p1, p3 - p1);
		glm::vec3 n1 = glm::cross(p2 - p, p3 - p);
		glm::vec3 n2 = glm::cross(p3 - p, p1 - p);
		glm::vec3 n3 = glm::cross(p1 - p, p2 - p);

		float sign1 = glm::dot(n1, N);
		if (sign1 < 0) {
			return hit;
		}

		float sign2 = glm::dot(n2, N);
		if (sign2 < 0) {
			return hit;
		}

		float sign3 = glm::dot(n3 ,N);
		if (sign3 < 0) {
			return hit;
		}
		
		hit.hit = true;
		hit.object = this;
		hit.intersection = transformationMatrix * glm::vec4(hit.intersection, 1.0);
		hit.normal = (normalMatrix * glm::vec4(hit.normal, 0.0));
		hit.normal = glm::normalize(hit.normal);
		hit.distance = glm::length(hit.intersection - ray.origin);
		
		return hit;
	}
};

struct Face {
    vector<int> vertices;
    vector<int> normals;
};

class Mesh : public Object {
private:
	vector<glm::vec3> vertices;
    vector<glm::vec3> normals;
    vector<Face> faces;
  vector<Triangle*> triangles;

public:
	Mesh(const string& filename, Material material){
		if (!loadFromFile(filename)) {
            throw std::runtime_error("Failed to load mesh from file: " + filename);
        }
		this->material = material;
	}

    bool loadFromFile(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Failed to open file: " << filename << endl;
            return false;
        }

        string line;
        while (getline(file, line)) {
            istringstream lineStream(line);
            string prefix;
            lineStream >> prefix;

            if (prefix == "v") {
                float x, y, z;
                lineStream >> x >> y >> z;
                vertices.emplace_back(x, y, z);

            } else if (prefix == "vn") {
                float nx, ny, nz;
                lineStream >> nx >> ny >> nz;
                normals.emplace_back(nx, ny, nz);

            } else if (prefix == "f") {
                Face face;
                string indexPair;

                while (lineStream >> indexPair) {
                    istringstream indexStream(indexPair);
          int v, vn;

          indexStream >> v;
          face.vertices.push_back(v - 1);

          if (indexStream.peek() == '/') {
            indexStream.ignore(2);
            indexStream >> vn;
            face.normals.push_back(vn - 1);
          }
        }
        faces.push_back(face);
      }
    }
    file.close();

    for (const auto& face : faces) {
      if (face.vertices.size() >= 3) {
        for (size_t i = 1; i + 1 <= face.vertices.size(); i++) {
          int v0_index = face.vertices[0];
          int v1_index = face.vertices[i - 1];
          int v2_index = face.vertices[i % face.vertices.size()];

          glm::vec3 v0 = vertices[v0_index];
          glm::vec3 v1 = vertices[v1_index];
          glm::vec3 v2 = vertices[v2_index];

          Triangle* triangle = new Triangle(this->material);

          // Assuming p1, p2, p3 are accessible in Triangle
          triangle->p1 = v0;
          triangle->p2 = v1;
          triangle->p3 = v2;

          triangle->setTransformation(this->transformationMatrix);

          triangles.push_back(triangle);
        }
      }
    }

    return true;
  }

  Hit intersect(Ray ray) {
    Hit closest_hit;
    closest_hit.hit = false;
    closest_hit.distance = INFINITY;

    glm::vec3 transformedOrigin = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0);
    glm::vec3 transformedDirection = inverseTransformationMatrix * glm::vec4(ray.direction, 0.0);
    transformedDirection = glm::normalize(transformedDirection);

    Ray transformedRay(transformedOrigin, transformedDirection);

    for (auto triangle : triangles) {
      Hit hit = triangle->intersect(transformedRay);
      if (hit.hit && hit.distance < closest_hit.distance) {
        closest_hit = hit;
        closest_hit.object = this;
      }
    }

    if (closest_hit.hit) {
      closest_hit.intersection = transformationMatrix * glm::vec4(closest_hit.intersection, 1.0);
      closest_hit.normal = glm::normalize(normalMatrix * glm::vec4(closest_hit.normal, 0.0));
      closest_hit.distance = glm::length(closest_hit.intersection - ray.origin);
    }

    return closest_hit;
  }
};

/**
 Light class
 */
class Light{
public:
  glm::vec3 position; ///< Position of the light source
  glm::vec3 color; ///< Color/intentisty of the light source
  Light(glm::vec3 position): position(position){
    color = glm::vec3(1.0);
  }
  Light(glm::vec3 position, glm::vec3 color): position(position), color(color){
  }
};

vector<Light *> lights; ///< A list of lights in the scene
//glm::vec3 ambient_light(0.1,0.1,0.1);
// new ambient light
glm::vec3 ambient_light(0.001,0.001,0.001);
vector<Object *> objects; ///< A list of all objects in the scene


/** Function for computing color of an object according to the Phong Model
 @param point A point belonging to the object for which the color is computer
 @param normal A normal vector the the point
 @param view_direction A normalized direction from the point to the viewer/camera
 @param material A material structure representing the material of the object
*/
glm::vec3 PhongModel(glm::vec3 point, glm::vec3 normal, glm::vec3 view_direction, Material material){

  glm::vec3 color(0.0);
  for(int light_num = 0; light_num < lights.size(); light_num++){

    glm::vec3 light_direction = glm::normalize(lights[light_num]->position - point);
    glm::vec3 reflected_direction = glm::reflect(-light_direction, normal);

    float NdotL = glm::clamp(glm::dot(normal, light_direction), 0.0f, 1.0f);
    float VdotR = glm::clamp(glm::dot(view_direction, reflected_direction), 0.0f, 1.0f);

    glm::vec3 diffuse_color = material.diffuse;
    glm::vec3 diffuse = diffuse_color * glm::vec3(NdotL);
    glm::vec3 specular = material.specular * glm::vec3(pow(VdotR, material.shininess));

    float r = glm::distance(point,lights[light_num]->position);
    r = max(r, 0.1f);
    color += lights[light_num]->color * (diffuse + specular) / r/r;
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
glm::vec3 trace_ray(Ray ray){

  Hit closest_hit;

  closest_hit.hit = false;
  closest_hit.distance = INFINITY;

  for(int k = 0; k<objects.size(); k++){
    Hit hit = objects[k]->intersect(ray);
    if(hit.hit == true && hit.distance < closest_hit.distance)
      closest_hit = hit;
  }

  glm::vec3 color(0.0);
  if(closest_hit.hit){
    color = PhongModel(closest_hit.intersection, closest_hit.normal, glm::normalize(-ray.direction), closest_hit.object->getMaterial());
  }else{
    color = glm::vec3(0.0, 0.0, 0.0);
	}
	return color;
}
/**
 Function defining the scene
 */
void sceneDefinition (){

	
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
	
	
	//Material green_diffuse;
	green_diffuse.ambient = glm::vec3(0.03f, 0.1f, 0.03f);
	green_diffuse.diffuse = glm::vec3(0.3f, 1.0f, 0.3f);

	//Material red_specular;
	red_specular.diffuse = glm::vec3(1.0f, 0.2f, 0.2f);
	red_specular.ambient = glm::vec3(0.01f, 0.02f, 0.02f);
	red_specular.specular = glm::vec3(0.5);
	red_specular.shininess = 10.0;

	//Material blue_specular;
	blue_specular.ambient = glm::vec3(0.02f, 0.02f, 0.1f);
	blue_specular.diffuse = glm::vec3(0.2f, 0.2f, 1.0f);
	blue_specular.specular = glm::vec3(0.6);
	blue_specular.shininess = 100.0;

  // Define material for the mesh
Material mesh_material;
mesh_material.ambient = glm::vec3(0.1f, 0.1f, 0.1f);
mesh_material.diffuse = glm::vec3(0.6f, 0.7f, 0.8f);
mesh_material.specular = glm::vec3(0.9f);
mesh_material.shininess = 50.0f;

// Create the mesh object
Mesh *mesh = new Mesh("/home/leonardo/dev/Uni/computer-graphics/Assignment 3/code/meshes/bunny.obj", mesh_material);

// Apply transformations if necessary
glm::mat4 translation = glm::translate(glm::vec3(0.0f, -1.0f, 5.0f));
glm::mat4 scaling = glm::scale(glm::vec3(1.0f, 1.0f, 1.0f));
mesh->setTransformation(translation * scaling);

// Add the mesh to the objects list
objects.push_back(mesh);

	
	lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(1.0, 1.0, 1.0)));
	lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.1)));
	lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4)));
	
    Material red_diffuse;
    red_diffuse.ambient = glm::vec3(0.09f, 0.06f, 0.06f);
    red_diffuse.diffuse = glm::vec3(0.9f, 0.6f, 0.6f);
        
    Material blue_diffuse;
    blue_diffuse.ambient = glm::vec3(0.06f, 0.06f, 0.09f);
    blue_diffuse.diffuse = glm::vec3(0.6f, 0.6f, 0.9f);
    objects.push_back(new Plane(glm::vec3(0,-3,0), glm::vec3(0.0,1,0)));
    objects.push_back(new Plane(glm::vec3(0,1,30), glm::vec3(0.0,0.0,-1.0), green_diffuse));
    objects.push_back(new Plane(glm::vec3(-15,1,0), glm::vec3(1.0,0.0,0.0), red_diffuse));
    objects.push_back(new Plane(glm::vec3(15,1,0), glm::vec3(-1.0,0.0,0.0), blue_diffuse));
    objects.push_back(new Plane(glm::vec3(0,27,0), glm::vec3(0.0,-1,0)));
    objects.push_back(new Plane(glm::vec3(0,1,-0.01), glm::vec3(0.0,0.0,1.0), green_diffuse));
	
	
	
}
glm::vec3 toneMapping(glm::vec3 intensity){
	float gamma = 1.0/2.0;
	float alpha = 12.0f;
	return glm::clamp(alpha * glm::pow(intensity, glm::vec3(gamma)), glm::vec3(0.0), glm::vec3(1.0));
}

void printProgress(float percentage) {
  int barWidth = 70; // Width of the progress bar

  std::cout << "[";
  int pos = barWidth * percentage;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos) std::cout << "▮";
    else std::cout << ".";
  }
  std::cout << "] " << int(percentage * 100.0) << " %\r";
  std::cout.flush();
}

int main(int argc, const char * argv[]) {

  clock_t t = clock(); // variable for keeping the time of the rendering

  int width = 480; //width of the image
  int height = 240; // height of the image
  float fov = 90; // field of view

  sceneDefinition(); // Let's define a scene

  Image image(width,height); // Create an image where we will store the result
  vector<glm::vec3> image_values(width*height);

  float s = 2*tan(0.5*fov/180*M_PI)/width;
  float X = -s * width / 2;
  float Y = s * height / 2;

  int totalPixels = width * height;
  int processed = 0;

  for(int i = 0; i < width ; i++)
    for(int j = 0; j < height ; j++){

      float dx = X + i*s + s/2;
      float dy = Y - j*s - s/2;
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
  cout<<"It took " << ((float)t)/CLOCKS_PER_SEC<< " seconds to render the image."<< endl;
  cout<<"I could render at "<< (float)CLOCKS_PER_SEC/((float)t) << " frames per second."<<endl;

  // Writing the final results of the rendering
  if (argc == 2){
    image.writeImage(argv[1]);
  }else{
    image.writeImage("./result.ppm");
  }


  return 0;
}
