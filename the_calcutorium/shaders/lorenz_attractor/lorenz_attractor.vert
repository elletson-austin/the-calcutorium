#version 330

in vec4 in_position;

uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 frag_pos;

void main() {
    frag_pos = in_position.xyz;
    gl_Position = u_proj * u_view * vec4(in_position.xyz, 1.0);
    gl_PointSize = 0.5; 
}
