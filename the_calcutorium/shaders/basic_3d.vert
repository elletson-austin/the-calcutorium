#version 330

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_color;

void main() {
    gl_Position = u_proj * u_view * vec4(in_position, 1.0);
    gl_PointSize = 2.0;
    v_color = in_color;
}
