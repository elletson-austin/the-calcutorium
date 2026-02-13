#version 330

in vec3 v_color;
out vec4 fragColor;
uniform float u_alpha_multiplier;

void main() {
    fragColor = vec4(v_color, 1.0 * u_alpha_multiplier);
}
