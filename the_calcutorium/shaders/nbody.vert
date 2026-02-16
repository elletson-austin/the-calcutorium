#version 330 core

in vec4 in_position;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_point_size;

void main() {
    gl_Position = u_proj * u_view * u_model * in_position;
    gl_PointSize = u_point_size;
}
