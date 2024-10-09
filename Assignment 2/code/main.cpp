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
	 Constructor of the ray
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

        // TODO Exercise 2 - Set the two remaining matrices
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
        // TODO uncommenting this fucks base of cone
        // works when used in global coordinate systems but not when in local cone coordinates
        glm::vec3 translated_point = point; // - ray.origin;

        // TODO Exercise 1 - Plane-ray intersection

        // This dot product tells us the relative alignment between the ray and the plane
        float alignmentToPlane  = glm::dot(ray.direction, normal);

        // The ray is parallel to the plane
        if (alignmentToPlane == 0) {
            return hit;
        }

        // This dot product tells us the projection of the vector from the ray’s origin to the plane onto the normal
        // vector. In other words how far the ray's origin is from the plane in the direction of the plane's normal
        float distanceToPlane = glm::dot(normal, translated_point - ray.origin);
        float intersectionFactor = distanceToPlane / alignmentToPlane;

        // The intersection occurs behind the camera
        if (intersectionFactor < 0) {
            return hit;
        }

        // Flip normal if its pointing away from camera
        glm::vec3 normal_facing_camera = normal;
        if (glm::dot(normal, ray.direction) > 0) {
            normal_facing_camera *= -1;
        }

        hit.hit = true;
        hit.intersection = ray.origin + intersectionFactor * ray.direction;
        hit.normal = glm::normalize(normal_facing_camera);
        hit.distance = glm::distance(ray.origin, hit.intersection);
        hit.object = this;

		return hit;
	}
};

class Cone : public Object{
private:
	Plane *plane;
	
public:
	Cone(Material material){
		this->material = material;
		plane = new Plane(glm::vec3(0.0f,1.0f,0.0f), glm::vec3(0.0f,1.0f,0.0f));
	}
	Hit intersect(Ray ray){
		
		Hit hit;
		hit.hit = false;

        // TODO Exercise 2
		/*  ---- Exercise 2 -----
		 * Implement the ray-cone intersection. Before intersecting the ray with the cone,
         * make sure that you transform the ray into the local coordinate system.
         * Remember about normalizing all the directions after transformations.
         *
		 * If the intersection is found, you have to set all the critical fields in the Hit structure
         * Remember that the final information about intersection point, normal vector and distance have to be given
         * in the global coordinate system.
         */

        // t = transformed
        glm::vec4 tRayOrigin = inverseTransformationMatrix * glm::vec4(ray.origin, 1.0);
        glm::vec4 tRayDirection = glm::normalize(inverseTransformationMatrix * glm::vec4(ray.direction, 0.0));

        /* a * t^2 + b * t^2 + c = 0
         * where
         * a = dx^2 + dz^2 - dy^2
         * b = 2(ox*dx + oz*dz - oy*dy)
         * c = ox^2 + oz^2 - oy^2
         *
         * o = ray origin
         * d = ray direction
         */
		float a = pow(tRayDirection.x, 2.0f) + pow(tRayDirection.z, 2.0f) - pow(tRayDirection.y, 2.0f);
        float b = 2 * (tRayOrigin.x * tRayDirection.x + tRayOrigin.z * tRayDirection.z - tRayOrigin.y * tRayDirection.y);
        float c = pow(tRayOrigin.x, 2.0f) + pow(tRayOrigin.z, 2.0f) - pow(tRayOrigin.y, 2.0f);
        float determinant = pow(b, 2.0f) - 4.0f * a * c;

        // Negative determinant means no solutions
        if (determinant < 0) {
            return hit;
        }

        float intersectionFactor = (-b - sqrt(determinant)) / (2 * a);
        if (intersectionFactor < 0) {
            return hit;
        }

        glm::vec4 tIntersectionPoint = tRayOrigin + intersectionFactor * tRayDirection;

        if (tIntersectionPoint.y < 0) {
            return hit;
        }

        if (tIntersectionPoint.y > 1) {
            Hit planeHit = plane->intersect(Ray(tRayOrigin, tRayDirection));
            // Intersection occurs outside the base of the cone
            if (!planeHit.hit || pow(planeHit.intersection.x, 2.0f) + pow(planeHit.intersection.z, 2.0f) >= 1.0f) {
                return hit;
            }
            hit.hit = true;
            hit.intersection = glm::vec3(transformationMatrix * glm::vec4(planeHit.intersection, 1.0f));
            hit.normal = glm::normalize(glm::vec3(normalMatrix * glm::vec4(planeHit.normal, 0.0f)));
            hit.distance = glm::distance(ray.origin, hit.intersection);
            hit.object = this;
            return hit;
        }

        glm::vec4 tNormal = glm::vec4(tIntersectionPoint.x, -tIntersectionPoint.y, tIntersectionPoint.z, 0.0);

        hit.hit = true;
        hit.intersection = glm::vec3(transformationMatrix * tIntersectionPoint);
        hit.normal = glm::normalize(glm::vec3(normalMatrix * tNormal));
        hit.distance = glm::distance(ray.origin, hit.intersection);
        hit.object = this;

		return hit;
	}
};

/**
 Light class
 */
class Light{
public:
	glm::vec3 position; ///< Position of the light source
	glm::vec3 color; ///< Color/intensity of the light source
	Light(glm::vec3 position): position(position){
		color = glm::vec3(1.0);
	}
	Light(glm::vec3 position, glm::vec3 color): position(position), color(color){
	}
};

vector<Light *> lights; ///< A list of lights in the scene
glm::vec3 ambient_light(0.1,0.1,0.1);
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

        // TODO Exercise 3
		/*  ---- Exercise 3-----
		
		 Include light attenuation due to the distance to the light source.
		 
		*/

        float lightDistance = glm::distance(lights[light_num]->position, point);
        float minimumDistance = 0.5f;
		color += lights[light_num]->color * (diffuse + specular) / pow(max(lightDistance, minimumDistance), 2.0f);
		
	
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

	/*  ---- All Exercises -----
	
	 Modify the scene definition according to the exercises
	 
	*/
	
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

    Material yellow_specular;
    yellow_specular.ambient = glm::vec3(1.0f, 1.0f, 0.0f);
    yellow_specular.diffuse = glm::vec3(1.0f, 1.0f, 0.0f);
    yellow_specular.specular = glm::vec3(0.6);
    yellow_specular.shininess = 100.0;

    objects.push_back(new Sphere(1.0, glm::vec3(1,-2,8), blue_specular));
	objects.push_back(new Sphere(0.5, glm::vec3(-1,-2.5,6), red_specular));
	// objects.push_back(new Sphere(1.0, glm::vec3(2,-2,6), green_diffuse));
		
	lights.push_back(new Light(glm::vec3(0, 26, 5), glm::vec3(0.4)));
	lights.push_back(new Light(glm::vec3(0, 1, 12), glm::vec3(0.4)));
	lights.push_back(new Light(glm::vec3(0, 5, 1), glm::vec3(0.4)));

    objects.push_back(new Plane(glm::vec3(-15, 0, 0),glm::vec3(1, 0, 0),red_specular));
    objects.push_back(new Plane(glm::vec3(15, 0, 0),glm::vec3(1, 0, 0),red_specular));
    objects.push_back(new Plane(glm::vec3(0, -3, 0),glm::vec3(0, 1, 0),blue_specular));
    objects.push_back(new Plane(glm::vec3(0, 27, 0),glm::vec3(0, 1, 0),blue_specular));
    objects.push_back(new Plane(glm::vec3(0, 0, -0.01),glm::vec3(0, 0, 1),green_diffuse));
    objects.push_back(new Plane(glm::vec3(0, 0, 30),glm::vec3(0, 0, 1),green_diffuse));

    Cone* yellowCone = new Cone(yellow_specular);
    glm::mat4 yellowTranslation = glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 9.0f, 14.0f));
    glm::mat4 yellowScaling = glm::scale(glm::mat4(1.0f), glm::vec3(3.0f, 12.0f, 3.0f));
    glm::mat4 yellowRotation = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    yellowCone->setTransformation(yellowTranslation * yellowRotation * yellowScaling);
    objects.push_back(yellowCone);

    Cone* greenCone = new Cone(green_diffuse);
    glm::mat4 greenTranslation = glm::translate(glm::mat4(1.0f), glm::vec3(6.0f, -3.0f, 7.0f));
    glm::mat4 greenScaling = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 3.0f, 1.0f));
    glm::mat4 greenRotation = glm::rotate(glm::mat4(1.0f), glm::radians(71.57f), glm::vec3(0.0f, 0.0f, 1.0f));
    greenCone->setTransformation(greenTranslation * greenRotation * greenScaling);
    objects.push_back(greenCone);
    // TODO changing order of transformations fucks the cone

}
glm::vec3 toneMapping(glm::vec3 intensity){

	/*  ---- Exercise 3-----
	
	 Implement a tone mapping strategy and gamma correction for a correct display.
	 
	*/
    float I_max = 1.0f;
    float c = 1.0f;

    intensity = glm::vec3(
            log(c * intensity.r + 1.0f) / log(I_max + 1.0f),
            log(c * intensity.g + 1.0f) / log(I_max + 1.0f),
            log(c * intensity.b + 1.0f) / log(I_max + 1.0f)
    );

    float gamma = 2.2f;
    intensity = glm::vec3(
            pow(intensity.r, 1.0f / gamma),
            pow(intensity.g, 1.0f / gamma),
            pow(intensity.b, 1.0f / gamma));
	
	return intensity;
}
int main(int argc, const char * argv[]) {

    clock_t t = clock(); // variable for keeping the time of the rendering

    int width = 1024; //width of the image
    int height = 768; // height of the image
    float fov = 90; // field of view

	sceneDefinition(); // Let's define a scene

	Image image(width,height); // Create an image where we will store the result

    float s = 2*tan(0.5*fov/180*M_PI)/width;
    float X = -s * width / 2;
    float Y = s * height / 2;

    for(int i = 0; i < width ; i++)
        for(int j = 0; j < height ; j++){

			float dx = X + i*s + s/2;
            float dy = Y - j*s - s/2;
            float dz = 1;

			glm::vec3 origin(0, 0, 0);
            glm::vec3 direction(dx, dy, dz);
            direction = glm::normalize(direction);

            Ray ray(origin, direction);

			image.setPixel(i, j, glm::clamp(toneMapping(trace_ray(ray)), glm::vec3(0.0), glm::vec3(1.0)));

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
