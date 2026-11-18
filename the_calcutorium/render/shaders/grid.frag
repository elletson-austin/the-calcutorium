#version 330

in vec3 v_color;
in float v_is_major;
out vec4 fragColor;
uniform float u_alpha_multiplier;

void main() {
    // Major gridlines are brighter and more opaque
    float alpha = 1.0 * u_alpha_multiplier;
    vec3 color = v_color;
    
    if (v_is_major > 0.5) {
        // Major gridline: brighter
        color = mix(v_color, vec3(1.0), 0.6);
        alpha *= 1.2;
    } else {
        // Minor gridline: dimmer
        alpha *= 0.6;
    }
    
    fragColor = vec4(color, min(alpha, 1.0));
}
