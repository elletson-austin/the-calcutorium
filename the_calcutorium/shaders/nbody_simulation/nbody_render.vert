#version 330 core

in vec3 in_position;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

void main() {
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
    gl_PointSize = 10.0; // Render bodies as points with a fixed size
}
