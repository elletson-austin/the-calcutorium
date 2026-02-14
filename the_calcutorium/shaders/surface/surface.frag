#version 330

in vec3 v_normal;
in vec3 v_pos;
in vec3 v_color;

out vec4 fragColor;

uniform vec3 u_light_pos;
uniform vec3 u_view_pos;

void main() {
    vec3 norm = normalize(v_normal);
    vec3 light_dir = normalize(u_light_pos - v_pos);
    
    // Ambient
    float ambient_strength = 0.2;
    vec3 ambient = ambient_strength * vec3(1.0, 1.0, 1.0);
    
    // Diffuse
    float diff = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
    
    // Specular
    float specular_strength = 0.4;
    vec3 view_dir = normalize(u_view_pos - v_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    vec3 specular = specular_strength * spec * vec3(1.0, 1.0, 1.0);
    
    vec3 result = (ambient + diffuse + specular) * v_color;
    fragColor = vec4(result, 1.0);
}
