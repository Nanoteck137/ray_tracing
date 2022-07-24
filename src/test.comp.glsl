#version 430

layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba8, set = 0, binding = 0) uniform image2D output_image;

layout(set = 0, binding = 1) uniform Data {
    mat4 view_matrix_inv;
    mat4 projection_matrix_inv;
};

struct Ray {
    vec3 origin;
    vec3 dir;
};

Ray createRay(vec3 origin, vec3 dir) {
    return Ray(origin, dir);
}

vec3 rayAt(Ray ray, float t) {
    return ray.origin + ray.dir * t;
}

bool hitSphere(Ray ray, vec3 position, float radius) {
    vec3 oc = ray.origin - position;
    float a = dot(ray.dir, ray.dir);
    float half_b = dot(oc, ray.dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = half_b * half_b - a * c;

    return discriminant > 0.0;
}

vec3 shootRay(Ray ray) {

    if(hitSphere(ray, vec3(4.0, 1.0, 0.0), 1.0)) {
        return vec3(1.0, 0.0, 0.0);
    }

    vec3 dir = normalize(ray.dir);
    float t = 0.5 * (dir.y + 1.0);

    vec3 color1 = vec3(1.0, 1.0, 1.0);
    vec3 color2 = vec3(0.5, 0.7, 1.0);

    return (1.0 - t) * color1 + t * color2;
}

void main() {
    ivec2 output_size = imageSize(output_image);

    if(gl_GlobalInvocationID.x >= output_size.x ||
        gl_GlobalInvocationID.y >= output_size.y) {
        return;
    }

    vec2 uv = vec2(gl_GlobalInvocationID.xy) / output_size;

    vec3 origin = (view_matrix_inv * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 dir = (projection_matrix_inv * vec4(uv, 0.0, 1.0)).xyz;
    dir = normalize(view_matrix_inv * vec4(dir, 0.0)).xyz;

    Ray ray = createRay(origin, dir);
    vec3 color = shootRay(ray);

    vec4 pixel_color = vec4(color, 1.0);

    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    imageStore(output_image, pixel_coords, pixel_color);
}
