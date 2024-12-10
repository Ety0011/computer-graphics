var vertexShaderCode =
    `#version 300 es
    in vec3 a_position;
    in vec3 a_color;
    in vec3 a_normal;

    out vec3 v_color;
    out vec3 v_normal;
    out vec3 v_position;

    uniform mat4 projection_matrix;
    uniform mat4 view_matrix;
    uniform mat4 model_matrix;

    void main(){
        v_color = a_color;

        v_position = vec3(model_matrix * vec4(a_position, 1.0));

        mat3 normal_matrix = transpose(inverse(mat3(model_matrix)));
        v_normal = normalize(normal_matrix * a_normal);

        gl_Position = projection_matrix * view_matrix * model_matrix * vec4(a_position,1.0);
    }`;

var fragmentShaderCode =
    `#version 300 es
    precision mediump float;

    in vec3 v_color;
    in vec3 v_normal;
    in vec3 v_position;

    out vec4 out_color;

    uniform vec3 light_direction;
    uniform vec3 light_color;
    uniform vec3 ambient_color;
    uniform float shininess;
    uniform vec3 view_position;

    void main(){
        vec3 normal = normalize(v_normal);
        vec3 light_dir = normalize(-light_direction);
        vec3 view_dir = normalize(view_position - v_position);

        float diff = max(dot(normal, light_dir), 0.0);

        vec3 reflection = reflect(-light_dir, normal);
        float spec = pow(max(dot(reflection, view_dir), 0.0), shininess);

        vec3 ambient = ambient_color * v_color;
        vec3 diffuse = diff * light_color * v_color;
        vec3 specular = spec * light_color; 

        vec3 result_color = ambient + diffuse + specular;
        out_color = vec4(result_color, 1.0);
    }`;

var gl; // WebGL context
var shader_program; // the GLSL program we will use for rendering
var plane_vao;
var cube_vao;
var sphere_vao;

// The function initilize the WebGL canvas
function initWebGL(){
    var canvas = document.getElementById("webgl-canvas");
    gl = canvas.getContext("webgl2");

    //keep the size of the canvas for leter rendering
    gl.viewportWidth = canvas.width;
    gl.viewportHeight = canvas.height;

    //check for errors
    if(gl){
        console.log("WebGL succesfully initialized.");
    }else{
        console.log("Failed to initialize WebGL.")
    }
}

// This function compiles a shader
function compileShader(shader, source, type, name = ""){
    // link the source of the shader to the shader object
    gl.shaderSource(shader,source);
    // compile the shader
    gl.compileShader(shader);
    // check for success and errors
    let success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if(success){
        console.log(name + " shader compiled succesfully.");
    }else{
        console.log(name + " vertex shader error.")
        console.log(gl.getShaderInfoLog(shader));
    }
}

// This function links the GLSL program by combining different shaders
function linkProgram(program,vertShader,fragShader){
    // attach vertex shader to the program
    gl.attachShader(program,vertShader);
    // attach fragment shader to the program
    gl.attachShader(program,fragShader);
    // link the program
    gl.linkProgram(program);
    // check for success and errors
    if (gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.log("The shaders are initialized.");
    }else{
        console.log("Could not initialize shaders.");
    }
}

function createGLSLPrograms(){
    var vertexShader = gl.createShader(gl.VERTEX_SHADER);
    compileShader(vertexShader, vertexShaderCode, gl.VERTEX_SHADER, "Vertex shader");
    // Creating fragment shader
    var fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    compileShader(fragmentShader, fragmentShaderCode, gl.FRAGMENT_SHADER, "Fragment shader");
    // Creating and linking the program
    shader_program = gl.createProgram();
    linkProgram(shader_program, vertexShader, fragmentShader);
}

function createVAO(vao, shader, vertices, colors, normals) {
    // Bind the VAO
    gl.bindVertexArray(vao);

    // Set up the vertex buffer
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

    // Link the vertex buffer to the shader attribute "a_position"
    const positionAttributeLocation = gl.getAttribLocation(shader, "a_position");
    gl.enableVertexAttribArray(positionAttributeLocation);
    gl.vertexAttribPointer(positionAttributeLocation, 3, gl.FLOAT, false, 0, 0);

    // Set up the color buffer
    const colorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);

    // Link the color buffer to the shader attribute "a_color"
    const colorAttributeLocation = gl.getAttribLocation(shader, "a_color");
    gl.enableVertexAttribArray(colorAttributeLocation);
    gl.vertexAttribPointer(colorAttributeLocation, 3, gl.FLOAT, false, 0, 0);

    // Set up the normals buffer
    const normalsBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, normalsBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

    // Link the normals buffer to the shader attribute "a_normals"
    const normalAttributeLocation = gl.getAttribLocation(shader, "a_normal");
    gl.enableVertexAttribArray(normalAttributeLocation);
    gl.vertexAttribPointer(normalAttributeLocation, 3, gl.FLOAT, false, 0, 0);

    // Unbind the VAO to ensure it is not accidentally modified
    gl.bindVertexArray(null);
}


function initBuffers() {
    plane_vao = gl.createVertexArray();
    createVAO(plane_vao, shader_program, plane_vertices, plane_colors, plane_normals);

    cube_vao = gl.createVertexArray();
    createVAO(cube_vao, shader_program, cube_vertices, cube_colors, cube_normals);

    sphere_vao = gl.createVertexArray();
    createVAO(sphere_vao, shader_program, sphere_vertices, sphere_colors, sphere_normals);
}


function draw(){
    // input variables for controling camera and light parameters
    // feel free to use these or create your own
    let camera_azimuthal_angle = document.getElementById("camera_azimuthal_angle").value / 360 * 2 * Math.PI;
    let camera_polar_angle = document.getElementById("camera_polar_angle").value / 360 * 2 * Math.PI;
    let camera_distance = document.getElementById("camera_distance").value / 10;
    let camera_fov = document.getElementById("camera_fov").value / 360 * 2 * Math.PI;
    let light_azimuthal_angle = document.getElementById("light_azimuthal_angle").value / 360 * 2 * Math.PI;
    let light_polar_angle = document.getElementById("light_polar_angle").value / 360 * 2 * Math.PI;

    // add computation of camera position
    let camera_x = camera_distance * Math.sin(camera_polar_angle) * Math.cos(camera_azimuthal_angle);
    let camera_y = camera_distance * Math.cos(camera_polar_angle);
    let camera_z = camera_distance * Math.sin(camera_polar_angle) * Math.sin(camera_azimuthal_angle);
    let camera_position = vec3.fromValues(camera_x, camera_y, camera_z);

    // add computation of light direction
    let light_x = Math.sin(light_polar_angle) * Math.cos(light_azimuthal_angle);
    let light_y = Math.cos(light_polar_angle);
    let light_z = Math.sin(light_polar_angle) * Math.sin(light_azimuthal_angle);
    let light_direction = vec3.fromValues(light_x, light_y, light_z);
    // you will need to use the above values to compute view and projection matrices

    gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
    gl.clearColor(0.2, 0.2, 0.2, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.CULL_FACE);
    gl.enable(gl.DEPTH_TEST);

    // enable the GLSL program for the rendering
    gl.useProgram(shader_program);

    // TIPS:
    // - Before drawing anything using the program you still have to set values of all uniforms.
    // - As long as you use the same shader program you do not need to set all uniforms everytime you draw new object. The programs remembers the uniforms after calling gl.drawArrays
    // - The same, if you draw the same object, e.g., cube, multiple times, you do not need to bind the corresponding VAO everytime you draw, but you may want to change the transformation matrices.

    // uniforms
    let light_direction_location = gl.getUniformLocation(shader_program, "light_direction");
    gl.uniform3fv(light_direction_location, light_direction);

    let light_color_location = gl.getUniformLocation(shader_program, "light_color");
    gl.uniform3fv(light_color_location, vec3.fromValues(0.5, 0.5, 0.5));

    let ambient_color_location = gl.getUniformLocation(shader_program, "ambient_color");
    gl.uniform3fv(ambient_color_location, vec3.fromValues(0.1, 0.1, 0.1));

    let shininess_location = gl.getUniformLocation(shader_program, "shininess");
    gl.uniform1f(shininess_location, 32);
    
    let view_position_location = gl.getUniformLocation(shader_program, "view_position");
    gl.uniform3fv(view_position_location, camera_position);

    // transformation pipeline matrices
    let projection_matrix_location = gl.getUniformLocation(shader_program, "projection_matrix");
    let projection_matrix = mat4.create();
    mat4.perspective(projection_matrix, camera_fov, gl.viewportWidth / gl.viewportHeight, 0.1, 100.0);
    gl.uniformMatrix4fv(projection_matrix_location, false, projection_matrix);

    let view_matrix_location = gl.getUniformLocation(shader_program, "view_matrix");
    let view_matrix = mat4.create();
    mat4.lookAt(view_matrix, camera_position, vec3.fromValues(0, 0, 0), vec3.fromValues(0, 1, 0));
    gl.uniformMatrix4fv(view_matrix_location, false, view_matrix);

    let model_matrix_location = gl.getUniformLocation(shader_program, "model_matrix");
    let model_matrix;

    // draw plane
    model_matrix = mat4.create();
    mat4.translate(model_matrix, model_matrix, vec3.fromValues(0, -0.5, 0));
    mat4.scale(model_matrix, model_matrix, vec3.fromValues(20, 20, 20));
    gl.uniformMatrix4fv(model_matrix_location, false, model_matrix);
    gl.bindVertexArray(plane_vao);
    gl.drawArrays(gl.TRIANGLES, 0, plane_vertices.length/3);

    // draw cube 1
    model_matrix = mat4.create();
    mat4.translate(model_matrix, model_matrix, vec3.fromValues(1.5, 0, 0));
    gl.uniformMatrix4fv(model_matrix_location, false, model_matrix);
    gl.bindVertexArray(cube_vao);
    gl.drawArrays(gl.TRIANGLES, 0, cube_vertices.length/3);

    // draw cube 2
    model_matrix = mat4.create();
    mat4.translate(model_matrix, model_matrix, vec3.fromValues(-1.5, 0, 0));
    gl.uniformMatrix4fv(model_matrix_location, false, model_matrix);
    gl.bindVertexArray(cube_vao);
    gl.drawArrays(gl.TRIANGLES, 0, cube_vertices.length/3);

    // draw sphere
    model_matrix = mat4.create();
    gl.uniformMatrix4fv(model_matrix_location, false, model_matrix);
    gl.bindVertexArray(sphere_vao);
    gl.drawArrays(gl.TRIANGLES, 0, sphere_vertices.length/3);

    // this line is required for creating an animation and updating the rendering
    window.requestAnimationFrame(function() {draw();});
}
function start(){
    // initialze WebGL
    initWebGL();
    // create GLSL programs
    createGLSLPrograms();
    // initialize all the buffers and set up the vertex array objects (VAO)
    initBuffers();
    // draw
    draw();
}