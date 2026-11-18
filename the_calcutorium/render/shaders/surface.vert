#version 330

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_color;

uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_model;

out vec3 v_normal;
out vec3 v_pos;
out vec3 v_color;

void main() {
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
    v_pos = (u_model * vec4(in_position, 1.0)).xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_color = in_color;
}
