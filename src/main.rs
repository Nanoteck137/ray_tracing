use std::time::Instant;
use std::collections::VecDeque;
use glam::f32::{ Vec3, Vec2 };

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

struct Camera {
    origin: Vec3,
    lower_left_corner: Vec3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    fn new() -> Self {
        let aspect_ratio = 16.0 / 9.0;
        let viewport_height = 2.0;
        let viewport_width = aspect_ratio * viewport_height;
        let focal_length = 1.0;

        let origin = Vec3::new(0.0, 0.0, 0.0);
        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

        Self {
            origin,
            lower_left_corner,
            horizontal,
            vertical
        }
    }

    fn get_ray(&self, uv: Vec2) -> Ray {
        let dir = self.lower_left_corner + uv.x * self.horizontal + uv.y * self.vertical - self.origin;
        Ray::new(self.origin, dir)
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

    material_id: usize,
}

struct Material {
    color: Vec3,
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
        record.normal = (record.point - self.position) / self.radius;
        record.material_id = self.material_id;

        true
    }
}

struct World {
    materials: Vec<Material>,
    spheres: Vec<Sphere>
}

impl World {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>
    {
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
            let target = record.point + record.normal + random_unit_vec3();
            let new_ray = Ray::new(record.point, target - record.point);

            let material = &self.materials[record.material_id];
            let color = material.color * self.shoot_ray(&new_ray, depth - 1);

            return color;
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

fn random_vec3_in_unit_shpere() -> Vec3 {
    loop {
        let v = random_vec3();
        if v.length_squared() >= 1.0 {
            continue;
        }

        return v;
    }
}

fn random_unit_vec3() -> Vec3 {
    random_vec3_in_unit_shpere().normalize()
}

struct TileJob {
    x: usize,
    y: usize,

    width: usize,
    height: usize,
}

fn main() {
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 1920;
    let image_height = (image_width as f32 / aspect_ratio) as usize;
    println!("Width: {} Height: {}", image_width, image_height);

    let samples_per_pixel = 30;
    let max_depth = 4;
    let camera = Camera::new();

    let mut materials = Vec::new();
    materials.push(Material {
        color: Vec3::new(1.0, 0.0, 1.0),
    });

    materials.push(Material {
        color: Vec3::new(0.3, 0.8, 0.3),
    });

    let mut spheres = Vec::new();
    spheres.push(Sphere {
        position: Vec3::new(0.0, 0.0, -1.0),
        radius: 0.5,

        material_id: 0,
    });

    spheres.push(Sphere {
        position: Vec3::new(0.0, -100.5, -1.0),
        radius: 100.0,

        material_id: 1,
    });

    let world = World {
        materials,
        spheres,
    };

    let mut image = bmp::Image::new(image_width as u32, image_height as u32);

    let mut job_queue = VecDeque::new();

    let tile_width = 32;
    let tile_height = 32;

    for tile_y in 0..(image_height / tile_height) {
        for tile_x in 0..(image_width / tile_width) {
            job_queue.push_back(TileJob {
                x: tile_x * tile_width,
                y: tile_y * tile_height,

                width: tile_width,
                height: tile_height,
            });
        }
    }

    let num_tiles_x = image_width / tile_width;
    let num_tiles_y = image_height / tile_height;

    if (image_height % tile_height) > 0{
        for i in 0..num_tiles_x {
            let tile_x = i;
            let tw = tile_width;
            let th = image_height % tile_height;

            let start_y = num_tiles_y * tile_height;

            job_queue.push_back(TileJob {
                x: tile_x * tw,
                y: start_y,

                width: tw,
                height: th,
            });
        }
    }

    if (image_width % tile_width) > 0{
        for i in 0..num_tiles_y {
            let tile_y = i;
            let tw = image_width % tile_width;
            let th = tile_height;

            let start_x = num_tiles_x * tile_width;

            job_queue.push_back(TileJob {
                x: start_x,
                y: tile_y * th,

                width: tw,
                height: th,
            });
        }
    }

    let tw = image_width % tile_width;
    let th = image_height % tile_height;

    if tw != 0 && th != 0 {
        let start_x = num_tiles_x * tile_width;
        let start_y = num_tiles_y * tile_width;

        job_queue.push_back(TileJob {
            x: start_x,
            y: start_y,

            width: tw,
            height: th,
        });
    }

    while !job_queue.is_empty() {
        let job = job_queue.pop_front().unwrap();

        for yoff in 0..job.height {
            for xoff in 0..job.width {
                let pixel_x = job.x + xoff;
                let pixel_y = job.y + yoff;

                let pixel_x = pixel_x as u32;
                let pixel_y = pixel_y as u32;

                let r = xoff as f32 / job.width as f32;
                let g = yoff as f32 / job.height as f32;

                let r = (r * 255.0) as u8;
                let g = (g * 255.0) as u8;
                let pixel = bmp::Pixel::new(r, g, 0);
                image.set_pixel(pixel_x, pixel_y, pixel);
            }
        }
    }

    image.save("result.bmp")
        .expect("Failed to save image");

    return;


    let now = Instant::now();

    for y in 0..image_height {
        let per = ((y as f32 / image_height as f32) * 100.0) as u32;
        print!("\rWorking: {}%", per);
        for x in 0..image_width {
            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);

            for sample in 0..samples_per_pixel {
                let u = (x as f32 + rand::random::<f32>()) / image_width as f32;
                let v = (y as f32 + rand::random::<f32>()) / image_height as f32;
                let uv = Vec2::new(u, v);

                let ray = camera.get_ray(uv);
                let color = world.shoot_ray(&ray, max_depth);
                pixel_color += color;
            }

            write_pixel_to_image(&mut image, image_width, image_height,
                                 x, y,
                                 pixel_color,
                                 samples_per_pixel);
        }
    }
    println!();

    let elapsed_time = now.elapsed();
    println!("Time: {:.2} s ({} ms)", elapsed_time.as_secs_f32(), elapsed_time.as_millis());

    image.save("result.bmp")
        .expect("Failed to save image");
}
