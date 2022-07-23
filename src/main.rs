use std::path::Path;
use glam::f32::{ Vec2, Vec3, Vec4, Mat4 };

const SAMPLES_PER_PIXEL: usize = 500;
const MAX_DEPTH: usize = 4;

mod cpu;

struct Ray {
    origin: Vec3,
    dir: Vec3,
}

impl Ray {
    fn new(origin: Vec3, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

#[derive(Clone)]
pub struct Camera {
    view_matrix_inv: Mat4,
    projection_matrix_inv: Mat4,
}

impl Camera {
    fn new(position: Vec3, look_at: Vec3, fov: f32, aspect_ratio: f32) -> Self {
        let view_matrix =
            Mat4::look_at_rh(position, look_at, Vec3::new(0.0, 1.0, 0.0));
        let proj_matrix =
            Mat4::perspective_rh(fov.to_radians(), aspect_ratio, 0.1, 100.0);

        Self {
            view_matrix_inv: view_matrix.inverse(),
            projection_matrix_inv: proj_matrix.inverse(),
        }
    }

    fn get_ray(&self, uv: Vec2) -> Ray {
        let origin = self.view_matrix_inv * Vec4::new(0.0, 0.0, 0.0, 1.0);
        let origin = origin.truncate();

        let direction =
            self.projection_matrix_inv * Vec4::from((uv, 0.0, 1.0));
        let direction = direction.truncate();
        let direction =
            self.view_matrix_inv * Vec4::from((direction, 0.0));
        let direction = direction.truncate();

        let ray_origin = origin;
        let ray_dir = direction.normalize();

        Ray::new(ray_origin, ray_dir)
    }
}

fn hit_sphere(sphere: &Sphere, ray: &Ray) -> f32 {
    let oc = ray.origin - sphere.position;
    let a = ray.dir.length_squared();
    let half_b = oc.dot(ray.dir);
    let c = oc.length_squared() - sphere.radius * sphere.radius;
    let discriminant = half_b * half_b - a * c;

    if discriminant < 0.0 {
        return -1.0;
    } else {
        return (-half_b - discriminant.sqrt()) / a;
    }
}

/*
fn ray_color(ray: &Ray) -> Vec3 {
    let t = hit_sphere(Vec3::new(0.0, 0.0, -1.0), 0.5, ray);
    if t > 0.0 {
        let normal = (ray.at(t) - Vec3::new(0.0, 0.0, -1.0)).normalize();
        return 0.5 * (normal + 1.0);
    }

    let t = hit_sphere(Vec3::new(0.0, -100.5, -1.0), 100.0, ray);
    if t > 0.0 {
        let normal = (ray.at(t) - Vec3::new(0.0, 0.0, -1.0)).normalize();
        return 0.5 * (normal + 1.0);
    }


    let dir = ray.dir.normalize();
    let t = 0.5 * (dir.y + 1.0);

    let color1 = Vec3::new(1.0, 1.0, 1.0);
    let color2 = Vec3::new(0.5, 0.7, 1.0);

    (1.0 - t) * color1 + t * color2
}
*/

fn write_pixel_to_image(image: &mut bmp::Image,
                        image_width: usize, image_height: usize,
                        x: usize, y: usize,
                        color: Vec3,
                        samples_per_pixel: usize)
{

    let r = color.x;
    let g = color.y;
    let b = color.z;

    let scale = 1.0 / samples_per_pixel as f32;

    let r = (r * scale).sqrt();
    let g = (g * scale).sqrt();
    let b = (b * scale).sqrt();

    let r = r.clamp(0.0, 0.999);
    let g = g.clamp(0.0, 0.999);
    let b = b.clamp(0.0, 0.999);

    let r = (r * 255.0) as u8;
    let g = (g * 255.0) as u8;
    let b = (b * 255.0) as u8;

    image.set_pixel(x as u32, (image_height - y - 1) as u32,
                    bmp::Pixel::new(r, g, b));
}

#[derive(Default)]
struct HitRecord {
    point: Vec3,
    normal: Vec3,
    t: f32,

    front_face: bool,

    material_id: usize,
}

impl HitRecord {
    fn set_face_normal(&mut self, ray: &Ray, outward_normal: Vec3) {
        self.front_face = ray.dir.dot(outward_normal) < 0.0;
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

struct Material {
    color: Vec3,

    metallic: bool,
    metallic_strength: f32,

    dielectric: bool,
    ir: f32,
}

struct Sphere {
    position: Vec3,
    radius: f32,

    material_id: usize,
}

impl Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32, record: &mut HitRecord)
        -> bool
    {
        let oc = ray.origin - self.position;
        let a = ray.dir.length_squared();
        let half_b = oc.dot(ray.dir);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            return false;
        }

        let sqrtd = discriminant.sqrt();

        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;

            if root < t_min || t_max < root {
                return false;
            }
        }

        record.t = root;
        record.point = ray.at(record.t);
        let normal = (record.point - self.position) / self.radius;
        record.set_face_normal(ray, normal);
        record.material_id = self.material_id;

        true
    }
}

fn reflectance(cosine: f32, ref_index: f32) -> f32 {
    let r0 = (1.0 - ref_index) / (1.0 + ref_index);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powf(5.0)
}

pub struct World {
    materials: Vec<Material>,
    spheres: Vec<Sphere>
}

impl World {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut closest = t_max;
        let mut current_record = None;

        for sphere in &self.spheres {
            let mut record = HitRecord::default();

            if sphere.hit(ray, t_min, closest, &mut record) {
                closest = record.t;
                current_record = Some(record);
            }
        }

        current_record
    }

    fn shoot_ray(&self, ray: &Ray, depth: usize) -> Vec3 {
        if depth <= 0 {
            return Vec3::new(0.0, 0.0, 0.0);
        }

        if let Some(record) = self.hit(ray, 0.001, f32::MAX) {
            let material = &self.materials[record.material_id];

            let mut final_color = material.color;

            if material.dielectric {
                let ir = material.ir;
                let refraction_ratio = if record.front_face {
                    1.0 / ir
                } else {
                    ir
                };

                let d = ray.dir.normalize();
                let cos_theta = ((-d).dot(record.normal)).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                let cannot_refract = (refraction_ratio * sin_theta) > 1.0;

                let dir = if cannot_refract || reflectance(cos_theta, refraction_ratio) > rand::random::<f32>() {
                    reflect(d, record.normal)
                } else {
                    refract(d, record.normal, refraction_ratio)
                };

                // let refracted = refract(d, record.normal, refraction_ratio);
                let new_ray = Ray::new(record.point, dir);

                let attenuation = Vec3::new(1.0, 1.0, 1.0);
                final_color =
                    attenuation * self.shoot_ray(&new_ray, depth - 1);
            }
            else if material.metallic {
                let reflected = reflect(ray.dir, record.normal);
                let fuzz = 1.0 - material.metallic_strength;
                let ray_dir = reflected + fuzz *
                    random_in_unit_sphere_vec3();
                let scattered_ray = Ray::new(record.point, ray_dir);

                let color = if reflected.dot(record.normal) > 0.0 {
                    let attenuation = material.color;
                    attenuation * self.shoot_ray(&scattered_ray, depth - 1)
                } else {
                    Vec3::new(0.0, 0.0, 0.0)
                };

                final_color *= color;
            } else {
                // let target = record.point + random_in_hemisphere(record.normal);
                // let new_ray = Ray::new(record.point, target - record.point);
                let dir = record.normal + random_unit_vec3();
                let new_ray = Ray::new(record.point, dir);

                let diffuse = self.shoot_ray(&new_ray, depth - 1);
                final_color *= diffuse;
            }

            return final_color;
        }

        let dir = ray.dir.normalize();
        let t = 0.5 * (dir.y + 1.0);

        let color1 = Vec3::new(1.0, 1.0, 1.0);
        let color2 = Vec3::new(0.5, 0.7, 1.0);

        (1.0 - t) * color1 + t * color2
    }
}

fn random_vec3() -> Vec3 {
    let x = rand::random::<f32>() * 2.0 - 1.0;
    let y = rand::random::<f32>() * 2.0 - 1.0;
    let z = rand::random::<f32>() * 2.0 - 1.0;
    Vec3::new(x, y, z)
}

fn random_in_unit_sphere_vec3() -> Vec3 {
    loop {
        let v = random_vec3();
        if v.length_squared() >= 1.0 {
            continue;
        }

        return v;
    }
}

fn random_unit_vec3() -> Vec3 {
    random_in_unit_sphere_vec3().normalize()
}

fn random_in_hemisphere(normal: Vec3) -> Vec3 {
    let unit_sphere = random_in_unit_sphere_vec3();
    if unit_sphere.dot(normal) > 0.0 {
        return unit_sphere;
    } else {
        return -unit_sphere;
    }
}

fn random_in_unit_disk() -> Vec3 {
    loop {
        let x = rand::random::<f32>() * 2.0 - 1.0;
        let y = rand::random::<f32>() * 2.0 - 1.0;
        let p = Vec3::new(x, y, 0.0);
        if p.length_squared() >= 1.0 {
            continue;
        }

        return p;
    }
}

fn reflect(input: Vec3, normal: Vec3) -> Vec3 {
    let dn = 2.0 * input.dot(normal);
    input - normal * dn
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Vec3 {
    let cos_theta = ((-uv).dot(n)).min(1.0);
    let out_perp = etai_over_etat * (uv + cos_theta * n);
    let out_parallel = -((1.0 - out_perp.length_squared()).abs()).sqrt() * n;

    out_perp + out_parallel
}


fn create_random_world() -> World {
    let mut materials = Vec::new();
    let mut spheres = Vec::new();

    let mut submit_material = |material: Material| -> usize {
        let id = materials.len();
        materials.push(material);

        id
    };

    for a in -4..4 {
        for b in -4..4 {
            let a = a as f32;
            let b = b as f32;

            let x = a + 0.9 * rand::random::<f32>();
            let y = 0.2;
            let z = b + 0.9 * rand::random::<f32>();
            let position = Vec3::new(x, y, z);

            let choose_mat = rand::random::<f32>();

            let material = if choose_mat < 0.8 {
                let color = random_vec3() * 0.9;
                submit_material(Material {
                    color,
                    metallic: false,
                    metallic_strength: 0.0,

                    dielectric: false,
                    ir: 0.0,
                })
            } else if choose_mat < 0.95 {
                let color = random_vec3();
                let strength = rand::random::<f32>();
                submit_material(Material {
                    color,
                    metallic: true,
                    metallic_strength: strength,

                    dielectric: false,
                    ir: 0.0,
                })
            } else {
                submit_material(Material {
                    color: Vec3::new(1.0, 1.0, 1.0),
                    metallic: false,
                    metallic_strength: 0.0,

                    dielectric: true,
                    ir: 1.5,
                })
            };

            spheres.push(Sphere {
                position,
                radius: 0.2,

                material_id: material,
            });
        }
    }

    // NOTE(patrik): Ground
    let ground_material = submit_material(Material {
        color: Vec3::new(0.5, 0.5, 0.5),
        metallic: false,
        metallic_strength: 0.0,

        dielectric: false,
        ir: 0.0,
    });

    spheres.push(Sphere {
        position: Vec3::new(0.0, -1000.0, 0.0),
        radius: 1000.0,

        material_id: ground_material,
    });

    // NOTE(patrik): Sphere 1

    let material = submit_material(Material {
        color: Vec3::new(0.1, 0.1, 0.6),
        metallic: false,
        metallic_strength: 0.0,

        dielectric: true,
        ir: 1.5,
    });

    spheres.push(Sphere {
        position: Vec3::new(0.0, 1.0, 0.0),
        radius: 1.0,

        material_id: material,
    });

    // NOTE(patrik): Sphere 2

    let material = submit_material(Material {
        color: Vec3::new(0.1, 0.1, 0.6),
        metallic: false,
        metallic_strength: 0.0,

        dielectric: false,
        ir: 0.0,
    });

    spheres.push(Sphere {
        position: Vec3::new(-4.0, 1.0, 0.0),
        radius: 1.0,

        material_id: material,
    });

    // NOTE(patrik): Sphere 3

    let material = submit_material(Material {
        color: Vec3::new(0.7, 0.6, 0.5),

        metallic: true,
        metallic_strength: 0.1,

        dielectric: false,
        ir: 0.0,
    });

    spheres.push(Sphere {
        position: Vec3::new(4.0, 1.0, 0.0),
        radius: 1.0,

        material_id: material,
    });

    World {
        materials,
        spheres,
    }
}

fn create_simple_world() -> World {
    let mut materials = Vec::new();
    let mut spheres = Vec::new();

    let mut submit_material = |material: Material| -> usize {
        let id = materials.len();
        materials.push(material);

        id
    };

    // NOTE(patrik): Ground
    let ground_material = submit_material(Material {
        color: Vec3::new(0.5, 0.5, 0.5),
        metallic: false,
        metallic_strength: 0.0,

        dielectric: false,
        ir: 0.0,
    });

    spheres.push(Sphere {
        position: Vec3::new(0.0, -1000.0, 0.0),
        radius: 1000.0,

        material_id: ground_material,
    });

    // NOTE(patrik): Sphere 3

    let material = submit_material(Material {
        color: Vec3::new(0.7, 0.6, 0.5),

        metallic: true,
        metallic_strength: 0.1,

        dielectric: false,
        ir: 0.0,
    });

    spheres.push(Sphere {
        position: Vec3::new(4.0, 1.0, 0.0),
        radius: 1.0,

        material_id: material,
    });

    World {
        materials,
        spheres,
    }
}

fn write_framebuffer_to_image<P>(path: P,
                                 image_width: usize,
                                 image_height: usize,
                                 framebuffer: &Vec<Vec3>)
    where P: AsRef<Path>
{
    let mut image = bmp::Image::new(image_width as u32, image_height as u32);

    for y in 0..image_height {
        for x in 0..image_width {

            let pixel = framebuffer[x + y * image_width];

            let r = (pixel.x * 255.0) as u8;
            let g = (pixel.y * 255.0) as u8;
            let b = (pixel.z * 255.0) as u8;

            let x = x as u32;
            let y = (image_height - y - 1) as u32;
            image.set_pixel(x, y, bmp::Pixel::new(r, g, b));
        }
    }

    image.save(path)
        .expect("Failed to save image");
}

fn main() {
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400;
    let image_height = (image_width as f32 / aspect_ratio) as usize;
    println!("Width: {} Height: {}", image_width, image_height);

    let position = Vec3::new(13.0, 2.0, 3.0);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    let fov = 20.0;
    let camera = Camera::new(position, look_at, fov, aspect_ratio);

    let world = create_random_world();
    // let world = create_simple_world();

    let tile_width = 64;
    let tile_height = 64;

    let job_queue = cpu::create_job_queue(image_width, image_height,
                                          tile_width, tile_height);
    let num_jobs = job_queue.len();

    let framebuffer = cpu::debug_write_job_queue_to_framebuffer(
        image_width,
        image_height,
        &job_queue);
    write_framebuffer_to_image("debug_tiles.bmp",
                               image_width,
                               image_height,
                               &framebuffer);


    let framebuffer =
        cpu::dispatch_work_cpu_singlethreaded(job_queue,
                                              image_width,
                                              image_height,
                                              MAX_DEPTH,
                                              SAMPLES_PER_PIXEL,
                                              camera,
                                              world);

    write_framebuffer_to_image("result.bmp",
                               image_width,
                               image_height,
                               &framebuffer);

    /*
    let framebuffer =
        cpu::dispatch_work_cpu_multithreaded(job_queue,
                                             image_width,
                                             image_height,
                                             MAX_DEPTH,
                                             SAMPLES_PER_PIXEL,
                                             camera,
                                             world);

    write_framebuffer_to_image("result.bmp",
                               image_width,
                               image_height,
                               &framebuffer);
    */

        // let framebuffer = dispatch_work_cpu_singlethreaded();
        // let framebuffer = dispatch_work_cpu_multithreaded();
        // let framebuffer = dispatch_work_gpu_compute_shader();
}
