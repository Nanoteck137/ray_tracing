#version 430

layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba8, set = 0, binding = 0) uniform image2D output_image;

layout(set = 0, binding = 1) uniform Data {
    mat4 view_matrix_inv;
    mat4 projection_matrix_inv;
    int num_spheres;
};

struct Sphere {
    vec3 position;
    float radius;
    int material_id;
};

struct Material {
    vec3 albedo;
};

layout(set = 0, binding = 2) readonly buffer spheres {
    Sphere spheres[];
} WorldSpheres;

layout(set = 0, binding = 3) readonly buffer materials {
    Material materials[];
} WorldMaterials;

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

struct HitRecord {
    vec3 point;
    vec3 normal;
    float t;
    bool front_face;
    int material_id;
};

bool hitSphere(Sphere sphere, Ray ray, float t_min, float t_max, out HitRecord record) {
    vec3 oc = ray.origin - sphere.position;
    float a = dot(ray.dir, ray.dir);
    float half_b = dot(oc, ray.dir);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = half_b * half_b - a * c;

    if(discriminant < 0.0) {
        return false;
    }

    float sqrtd = sqrt(discriminant);

    float root = (-half_b - sqrtd) / a;
    if(root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;

        if(root < t_min || t_max < root) {
            return false;
        }
    }

    float t = root;
    vec3 point = rayAt(ray, t);
    vec3 normal = (point - sphere.position) / sphere.radius;

    record.t = t;
    record.point = point;
    record.normal = normal;
    record.material_id = sphere.material_id;

    return true;
}

/*
fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
    self.front_face = ray.dir.dot(outward_normal) < 0.0;
    self.normal = if self.front_face {
        outward_normal
    } else {
        -outward_normal
    };
}
*/

bool hit(Ray ray, float t_min, float t_max, out HitRecord best) {
    float closest = t_max;
    bool foundHit = false;

    for(int i = 0; i < num_spheres; i++) {
        Sphere sphere = WorldSpheres.spheres[i];

        HitRecord record;
        if(hitSphere(sphere, ray, t_min, closest, record) == true) {
            closest = record.t;
            best = record;
            foundHit = true;
        }
    }

    return foundHit;
}

uint rng_state = 1231893;

uint rand_xorshift() {
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

float random() {
    return float(rand_xorshift()) * (1.0 / 4294967296.0);
}

vec3 random_vec3() {
    return vec3(random(), random(), random());
}

vec3 random_in_unit_sphere_vec3() {
    while(true) {
        vec3 v = random_vec3();
        if(dot(v, v) >= 1.0) {
            continue;
        }

        return v;
    }
}

vec3 random_unit_vec3() {
    return normalize(random_in_unit_sphere_vec3());
}

vec3 shootRay(Ray ray) {
    HitRecord record;
    if(hit(ray, 0.01, 1.0 / 0.0, record)) {
        Material material = WorldMaterials.materials[record.material_id];
        return material.albedo;
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

    uint test = uint(uv.x) + uint(uv.y) * uint(output_size.x);
    rng_state = test;

    uv = uv * 2.0 - 1.0;


    vec3 origin = (view_matrix_inv * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 dir = (projection_matrix_inv * vec4(uv, 0.0, 1.0)).xyz;
    dir = normalize(view_matrix_inv * vec4(dir, 0.0)).xyz;

    Ray ray = createRay(origin, dir);
    vec3 color = shootRay(ray);

    vec4 pixel_color = vec4(color, 1.0) * vec4(random_vec3(), 1.0);

    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    imageStore(output_image, pixel_coords, pixel_color);
}
