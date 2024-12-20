// -------------Shaders for the terrain rendering----------------
var vertexTerrainShaderCode =
`#version 300 es

in vec3 a_position;

out float v_height;
out vec2 v_texCoord;
out vec3 v_normal;
out vec3 v_position;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

uniform sampler2D u_heightmap;

void main() {
    vec2 texCoord = vec2(a_position.x + 0.5, a_position.z + 0.5);

    float heightLeft = texture(u_heightmap, texCoord + vec2(-0.01, 0.0)).r;
    float heightRight = texture(u_heightmap, texCoord + vec2(0.01, 0.0)).r;
    float heightDown = texture(u_heightmap, texCoord + vec2(0.0, -0.01)).r;
    float heightUp = texture(u_heightmap, texCoord + vec2(0.0, 0.01)).r;

    vec3 du = vec3(0.02, heightRight - heightLeft, 0.0);
    vec3 dv = vec3(0.0, heightUp - heightDown, 0.02);

    vec3 normal = normalize(cross(du, dv));

    mat3 normal_matrix = transpose(inverse(mat3(modelMatrix)));

    float height = texture(u_heightmap, texCoord).r;
    if (height <= 0.05) {
        height = 0.05;
    }

    v_normal = normalize(normal_matrix * normal);
    v_height = height;
    v_texCoord = texCoord;
    v_position = vec3(modelMatrix * vec4(a_position.x, height * 0.3, a_position.z, 1.0));
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(a_position.x, height * 0.3, a_position.z, 1.0);
}`;

var fragmentTerrainShaderCode =
`#version 300 es
precision highp float;

in float v_height;
in vec2 v_texCoord;
in vec3 v_normal;
in vec3 v_position;

out vec4 out_color;

uniform sampler2D u_grassTexture;
uniform sampler2D u_sandTexture;
uniform sampler2D u_graniteTexture;
uniform vec3 lightDirection;
uniform vec3 viewPosition;

const vec3 lightColor = vec3(1.0,1.0,1.0);
const float ambientCoeff = 0.05;
const float diffuseCoeff = 0.8;
const float specularCoeff = 0.01;
const float shininessCoeff = 1.0;

const float gamma = 1.8;

void main() {
    vec3 sandColor = texture(u_sandTexture, v_texCoord).xyz;
    vec3 grassColor = texture(u_grassTexture, v_texCoord).xyz;
    vec3 graniteColor = texture(u_graniteTexture, v_texCoord).xyz;
    vec3 whiteColor = vec3(1.0);
    vec3 waterColor = vec3(0.0, 0.0, 1.0);

    vec3 color;
    if (v_height <= 0.05885) {
        color = waterColor;
    } else if (v_height < 0.06) {
        float blend = smoothstep(0.05, 0.06, v_height);
        color = mix(waterColor, sandColor, blend);
    } else if (v_height < 0.1) {
        float blend = smoothstep(0.06, 0.1, v_height);
        color = mix(sandColor, grassColor, blend);
    } else if (v_height < 0.3) {
        float blend = smoothstep(0.1, 0.3, v_height);
        color = mix(grassColor, graniteColor, blend);
    } else {
        float blend = smoothstep(0.3, 0.4, v_height);
        color = mix(graniteColor, whiteColor, blend);
    }

    vec3 lightDirection = normalize(-lightDirection);
    vec3 viewDirection = normalize(viewPosition - v_position);

    vec3 V = normalize(viewDirection);
    vec3 N = normalize(v_normal);
    vec3 L = normalize(lightDirection);
    vec3 R = normalize(reflect(-L,N));

    vec3 ambient = ambientCoeff * color;
    vec3 diffuse = vec3(diffuseCoeff) * lightColor * color * vec3(max(dot(N,L), 0.0));
    vec3 specular = vec3(specularCoeff) * vec3(pow(max(dot(R,V), 0.0), shininessCoeff));

    color = ambient + diffuse + specular;
    color = pow(color,vec3(1.0/gamma));
    out_color = vec4(color, 1.0);
}`;

var vertexShaderCode =
`#version 300 es
in vec3 a_position;
in vec3 a_color;
in vec3 a_normal;

out vec3 v_color;
out vec3 v_normal;
out vec3 v_lightDirection;
out vec3 v_viewDirection;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 lightDirection;

void main(){
    v_color = a_color;
    v_normal = vec3(viewMatrix * modelMatrix * vec4(a_normal,0.0));
    v_lightDirection = vec3(viewMatrix * vec4(lightDirection, 0.0));
    v_viewDirection  = -vec3(viewMatrix * modelMatrix * vec4(a_position,1.0)); // in the eye space the camera is in (0,0,0)!
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(a_position,1.0);
}`;

var fragmentShaderCode =
`#version 300 es
precision highp float;

in vec3 v_color;
in vec3 v_viewDirection;
in vec3 v_normal;
in vec3 v_lightDirection;

const vec3 lightColor = vec3(1.0,1.0,1.0);
const float ambientCoeff = 0.05;
const float diffuseCoeff = 0.5;
const float specularCoeff = 0.5;
const float shininessCoeff = 50.0;

const float gamma = 1.8;

out vec4 out_color;
void main(){

    vec3 V = normalize(v_viewDirection);
    vec3 N = normalize(v_normal);
    vec3 L = normalize(v_lightDirection.xyz);
    vec3 R = normalize(reflect(-L,N));

    vec3 ambient = ambientCoeff * v_color;
    vec3 diffuse = vec3(diffuseCoeff) * lightColor * v_color * vec3(max(dot(N,L), 0.0));
    vec3 specular = vec3(specularCoeff) * vec3(pow(max(dot(R,V), 0.0), shininessCoeff));

    vec3 color = ambient + diffuse + specular;
    color = pow(color,vec3(1.0/gamma));
    out_color = vec4(color, 1.0);
}`;


var gl; // WebGL context
var shaderProgram; // the GLSL program we will use for rendering
var cube_vao; // the vertex array object for the cube
var sphere_vao; // the vertex array object for the sphere
var plane_vao; // the vertex array object for the plane

var terrain_vao; // the vertex array object for the terrain
var terrainShaderProgram; // shader program for rendering the terrain


function createGLSLProgram(program, vertCode, fragCode){
    let vertexShader = gl.createShader(gl.VERTEX_SHADER);
    compileShader(vertexShader, vertCode, gl.VERTEX_SHADER, "Vertex shader");
    // Creating fragment shader
    let fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    compileShader(fragmentShader, fragCode, gl.FRAGMENT_SHADER, "Fragment shader");
    // Creating and linking the program
    linkProgram(program, vertexShader, fragmentShader);
}

function createGLSLPrograms(){
    shaderProgram = gl.createProgram();
    createGLSLProgram(shaderProgram, vertexShaderCode, fragmentShaderCode);

    //------------- Creating shader program for the terrain rendering ---------------
    terrainShaderProgram = gl.createProgram();
    createGLSLProgram(terrainShaderProgram, vertexTerrainShaderCode, fragmentTerrainShaderCode);
}

function createVAO(vao, shader, vertices, normals, colors){
    // a buffer for vertices
    let vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

    // a buffer for color
    let colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

    // a buffer for normals
    let normalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

    // bind VAO
    gl.bindVertexArray(vao);

    // position attributes
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    let positionAttributeLocation = gl.getAttribLocation(shader, "a_position");
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.vertexAttribPointer(positionAttributeLocation, 3, gl.FLOAT, false, 0, 0);

    // color attributes
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    let colorAttributeLocation = gl.getAttribLocation(shader, "a_color");
    gl.enableVertexAttribArray(colorAttributeLocation);
    gl.vertexAttribPointer(colorAttributeLocation, 3, gl.FLOAT, false, 0, 0);

    // normal attributes
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    let normalAttributeLocation = gl.getAttribLocation(shader, "a_normal");
    gl.enableVertexAttribArray(normalAttributeLocation);
    gl.vertexAttribPointer(normalAttributeLocation, 3, gl.FLOAT, false, 0, 0);
}

function initBuffers(){
    cube_vao = gl.createVertexArray();
    createVAO(cube_vao, shaderProgram, cube_vertices, cube_normals, cube_colors);
    sphere_vao = gl.createVertexArray();
    createVAO(sphere_vao, shaderProgram, sphere_vertices, sphere_vertices, sphere_colors);
    plane_vao = gl.createVertexArray();
    createVAO(plane_vao, shaderProgram, plane_vertices, plane_normals, plane_colors);


    //------------- Creating VBO and VAO for terrain ---------------

    //buffer for the terrain_vaovar normalBuffer = gl.createBuffer();
    let terrainVertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, terrainVertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(terrain_vertices), gl.STATIC_DRAW);
    gl.vertexAttribPointer(terrainVertexBuffer, 3, gl.FLOAT, false, 0, 0);

    // Creating VAO for the terrain
    terrain_vao = gl.createVertexArray();
    gl.bindVertexArray(terrain_vao);

    let positionAttributeLocation = gl.getAttribLocation(terrainShaderProgram, "a_position");
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.vertexAttribPointer(positionAttributeLocation, 3, gl.FLOAT, false, 0, 0);
}

function draw(){
    let camera_azimuthal_angle = document.getElementById("camera_azimuthal_angle").value / 360 * 2 * Math.PI;
    let camera_polar_angle = document.getElementById("camera_polar_angle").value / 360 * 2 * Math.PI;
    let camera_distance = document.getElementById("camera_distance").value / 10;
    let camera_fov = document.getElementById("camera_fov").value / 360 * 2 * Math.PI;
    let light_azimuthal_angle = document.getElementById("light_azimuthal_angle").value / 360 * 2 * Math.PI;
    let light_polar_angle = document.getElementById("light_polar_angle").value / 360 * 2 * Math.PI;

    // computing the camera position from the angles
    let camera_x = camera_distance * Math.sin(camera_polar_angle) * Math.cos(camera_azimuthal_angle);
    let camera_y = camera_distance * Math.cos(camera_polar_angle);
    let camera_z = -camera_distance * Math.sin(camera_polar_angle) * Math.sin(camera_azimuthal_angle);
    let cameraPosition = vec3.fromValues(camera_x, camera_y, camera_z);

    // computing the light direction from the angles
    let light_x = Math.sin(light_polar_angle) * Math.cos(light_azimuthal_angle);
    let light_y = Math.cos(light_polar_angle);
    let light_z = -Math.sin(light_polar_angle) * Math.sin(light_azimuthal_angle);
    let lightDirection = vec3.fromValues(light_x, light_y, light_z);

    // view matrix
    let viewMatrix = mat4.create();
    mat4.lookAt(viewMatrix, cameraPosition, vec3.fromValues(0,0,0), vec3.fromValues(0,1,0));
    // projection matrix
    let projectionMatrix = mat4.create();
    mat4.perspective(projectionMatrix, camera_fov, 1.0, 0.1, 40.0);

    // model matrix (only definition, the value will be set when drawing a specific object)
    let modelMatrix = mat4.create();

    // set the size of our rendering area
    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);

    // setting the background color and clearing the color buffer
    gl.clearColor(0.2, 0.2, 0.2, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.enable(gl.CULL_FACE);
    gl.enable(gl.DEPTH_TEST);

    // enable the GLSL program for the rendering
    gl.useProgram(shaderProgram);

    // getting the locations of uniforms
    let modelMatrixLocation = gl.getUniformLocation(shaderProgram,"modelMatrix");
    let viewMatrixLocation = gl.getUniformLocation(shaderProgram,"viewMatrix");
    let projectionMatrixLocation = gl.getUniformLocation(shaderProgram,"projectionMatrix");
    let lightDirectionLocation = gl.getUniformLocation(shaderProgram,"lightDirection");

    // setting the uniforms which are common for the entires scene
    gl.uniformMatrix4fv(viewMatrixLocation, false, viewMatrix);
    gl.uniformMatrix4fv(projectionMatrixLocation, false, projectionMatrix);
    gl.uniform3fv(lightDirectionLocation, lightDirection);

    //drawing the sphere
    gl.bindVertexArray(sphere_vao);
    mat4.fromTranslation(modelMatrix,vec3.fromValues(0.0, 0.0, 0.0));
    gl.uniformMatrix4fv(modelMatrixLocation, false, modelMatrix);
    gl.drawArrays(gl.TRIANGLES, 0, sphere_vertices.length/3);

    //-----------------------------------------------
    //---------- Drawing the terrain-----------------
    //-----------------------------------------------

    // You have to start using the new shader program for terrain rendering.
    // Remember to pass all the matrices and the illumination information
    // Remember to get first all the locations of the uniforms in the new GLSL program
    // and then set up the values their values.
    // Note that the code for setting up the textures
    // is already provided below.

    gl.useProgram(terrainShaderProgram);

    // getting the locations of uniforms
    let tModelMatrixLocation = gl.getUniformLocation(terrainShaderProgram,"modelMatrix");
    let tViewMatrixLocation = gl.getUniformLocation(terrainShaderProgram,"viewMatrix");
    let tProjectionMatrixLocation = gl.getUniformLocation(terrainShaderProgram,"projectionMatrix");
    let tLightDirectionLocation = gl.getUniformLocation(terrainShaderProgram,"lightDirection");
    let viewPositionLocation = gl.getUniformLocation(terrainShaderProgram, "viewPosition");

    let tModelMatrix = mat4.create();
    let scaleFactor = 20.0;
    mat4.fromTranslation(tModelMatrix, vec3.fromValues(0.0, -5.0, 0.0));
    mat4.scale(tModelMatrix, tModelMatrix, vec3.fromValues(scaleFactor, scaleFactor, scaleFactor));

    let tViewMatrix = mat4.create();
    mat4.lookAt(tViewMatrix, cameraPosition, vec3.fromValues(0,0,0), vec3.fromValues(0,1,0));

    let tProjectionMatrix = mat4.create();
    mat4.perspective(tProjectionMatrix, camera_fov, 1.0, 0.1, 40.0);
    
    gl.uniformMatrix4fv(tModelMatrixLocation, false, tModelMatrix);
    gl.uniformMatrix4fv(tViewMatrixLocation, false, viewMatrix);
    gl.uniformMatrix4fv(tProjectionMatrixLocation, false, projectionMatrix);
    gl.uniform3fv(tLightDirectionLocation, lightDirection);
    gl.uniform3fv(viewPositionLocation, cameraPosition);
    
    for (let i = 0; i < terrainTextures.length; i++){
       let textureLocation = gl.getUniformLocation(terrainShaderProgram, terrainTextures[i].uniformName);
       gl.activeTexture(gl.TEXTURE0 + i);
       gl.bindTexture(gl.TEXTURE_2D, terrainTextures[i].glTexture);
       gl.uniform1i(textureLocation, i);
    }

    gl.bindVertexArray(terrain_vao);
    gl.drawArrays(gl.TRIANGLES, 0, terrain_vertices.length/3);

    window.requestAnimationFrame(function() {draw();});
}

// The function below creates textures and sets default parameters
// Feel free to play around with them to see how your rendering changes
function createTextures(){
    for (let texture of terrainTextures) {
        texture.glTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture.glTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texture);
        gl.generateMipmap(gl.TEXTURE_2D);
    }
    
}
function start(){

    initWebGL();
    createGLSLPrograms();
    createTextures(); // creating textures on GPU
    initBuffers();
    draw();
    }

    var leftToRead; // variable for counting all the textures that were already read from the files
    var terrainTextures = []; // array for storing all the texture information; it does not need to be changed

    // a list of the paths to the files with textures
    // add here the paths to the files from which the textures should be read
    var textureFiles = [
        "./Lugano.png",
        "./grass.jpg",
        "./sand.jpg",
        "./granite.jpg"
    ];

    // textureVariables should contain the names of uniforms in the shader program
    // IMPORTAN: if you are going to use the code we provide,
    // make sure the names below are identical to the one you use in the shader program
    var textureVariables = [
        "u_heightmap",
        "u_grassTexture",
        "u_sandTexture",
        "u_graniteTexture"
    ];

    function count_down(){
    leftToRead = leftToRead - 1;
    if (leftToRead == 0){
        start();
    }
}

function main(){

    // Loading the textures
    leftToRead = textureFiles.length;
    if(leftToRead == 0){
        start();
    }else{
        for(let i = 0; i < textureFiles.length; i++){
            terrainTextures.push(new Image());
            terrainTextures[i].src = textureFiles[i];
            terrainTextures[i].onload = count_down;
            terrainTextures[i].uniformName = textureVariables[i];
        }
    }
}