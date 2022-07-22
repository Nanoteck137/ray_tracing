use std::time::{ Instant, Duration };
use std::collections::VecDeque;
use std::sync::{ Arc, Mutex, RwLock };
use glam::f32::{ Vec2, Vec3, Vec4, Mat4 };

const SAMPLES_PER_PIXEL: usize = 500;
const MAX_DEPTH: usize = 4;

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
struct Camera {
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

struct World {
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

struct TileJob {
    x: usize,
    y: usize,

    width: usize,
    height: usize,
}

struct JobResult {
    x: usize,
    y: usize,

    width: usize,
    height: usize,

    pixel_colors: Vec<Vec3>,
}

fn execute_job(job: &TileJob,
               image_width: usize,
               image_height: usize,
               camera: &Camera,
               world: &World)
    -> JobResult
{
    let size = job.width * job.height;
    let mut result = Vec::with_capacity(size);

    for yoff in 0..job.height {
        for xoff in 0..job.width {
            let x = job.x + xoff;
            let y = job.y + yoff;

            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);
            for _sample in 0..SAMPLES_PER_PIXEL {
                let u = (x as f32 + rand::random::<f32>()) / image_width as f32;
                let v = (y as f32 + rand::random::<f32>()) / image_height as f32;
                let u = u * 2.0 - 1.0;
                let v = v * 2.0 - 1.0;
                let uv = Vec2::new(u, v);

                let ray = camera.get_ray(uv);
                let color = world.shoot_ray(&ray, MAX_DEPTH);
                pixel_color += color;
            }

            result.push(pixel_color);
        }
    }

    JobResult {
        x: job.x,
        y: job.y,

        width: job.width,
        height: job.height,

        pixel_colors: result,
    }
}

struct WorkerData {
    thread_id: usize,

    image_width: usize,
    image_height: usize,
    world: Arc<RwLock<World>>,
    camera: Camera,

    job_queue: Arc<Mutex<VecDeque<TileJob>>>,
    job_results: Arc<Mutex<Vec<JobResult>>>,
}

fn worker(data: WorkerData)
{
    loop {
        let job = {
            let mut lock = data.job_queue.lock().unwrap();
            lock.pop_front()
        };

        if let Some(job) = job {
            let world_lock = data.world.read().unwrap();
            let result = execute_job(&job,
                                     data.image_width,
                                     data.image_height,
                                     &data.camera,
                                     &world_lock);
            {
                let mut lock = data.job_results.lock().unwrap();
                lock.push(result);
            }
        } else {
            // NOTE(patrik): Queue empty
            break;
        }
    }
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

fn create_job_queue(image_width: usize,
                    image_height: usize,
                    tile_width: usize,
                    tile_height: usize)
    -> VecDeque<TileJob>
{
    let mut job_queue = VecDeque::new();

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

    job_queue
}

fn debug_job_queue(image: &mut bmp::Image, mut job_queue: VecDeque<TileJob>) {
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
    let mut image = bmp::Image::new(image_width as u32, image_height as u32);

    let tile_width = 64;
    let tile_height = 64;

    let job_queue = create_job_queue(image_width, image_height,
                                     tile_width, tile_height);
    let num_jobs = job_queue.len();

    // debug_job_queue(&mut image, job_queue);

    /*
    */

    // Thread:
    //   Look for job
    //   Execute job
    //   Send the result

    let job_queue = Arc::new(Mutex::new(job_queue));
    let job_results = Arc::new(Mutex::new(Vec::new()));

    let world = Arc::new(RwLock::new(world));

    // 1 thread(s) (debug) : Time: 19.78 s (19777 ms)
    // 4 thread(s) (debug) : Time: 5.28 s (5279 ms)
    // 8 thread(s) (debug) : Time: 4.11 s (4106 ms)
    //
    // 1 thread(s) (release) : Time: 0.89 s (894 ms)
    // 4 thread(s) (release) : Time: 0.23 s (233 ms)
    // 8 thread(s) (release) : Time: 0.17 s (166 ms)
    //
    // Time: 27.61 s (27607 ms)
    // Time: 19.22 s (19224 ms)

    let num_threads = 8;
    let mut thread_join_handles = Vec::with_capacity(num_threads);
    for thread_id in 0..num_threads {
        let data = WorkerData {
            thread_id,

            image_width,
            image_height,

            world: world.clone(),
            camera: camera.clone(),

            job_queue: job_queue.clone(),
            job_results: job_results.clone(),
        };

        let join_handle = std::thread::spawn(move || {
            worker(data);
        });

        thread_join_handles.push(join_handle);
    }

    let now = Instant::now();

    /*
    let mut lock = job_queue.lock().unwrap();
    while !lock.is_empty() {
        let job = lock.pop_front().unwrap();

        let result = execute_job(&job,
                                 image_width,
                                 image_height,
                                 &camera,
                                 &world);

        for yoff in 0..result.height {
            for xoff in 0..result.width {
                let x = result.x + xoff;
                let y = result.y + yoff;
                let color = result.pixel_colors[xoff + yoff * result.width];
                write_pixel_to_image(&mut image,
                                     image_width,
                                     image_height,
                                     x, y,
                                     color,
                                     SAMPLES_PER_PIXEL)
            }
        }
    }
    */

    loop {
        let current_finished_jobs = {
            let lock = job_results.lock().unwrap();
            lock.len()
        };

        let per = current_finished_jobs as f32 / num_jobs as f32;
        print!("\rFinished Jobs: {:.2}", per * 100.0);
        use std::io::Write;
        std::io::stdout().flush().unwrap();

        if current_finished_jobs >= num_jobs {
            break;
        }

        std::thread::sleep(Duration::from_millis(500));
    }

    println!();

    for handle in thread_join_handles {
        handle.join()
            .expect("Failed to join thread");
    }

    let lock = job_results.lock().unwrap();

    for result in lock.iter() {
        for yoff in 0..result.height {
            for xoff in 0..result.width {
                let x = result.x + xoff;
                let y = result.y + yoff;
                let color = result.pixel_colors[xoff + yoff * result.width];
                write_pixel_to_image(&mut image,
                                     image_width,
                                     image_height,
                                     x, y,
                                     color,
                                     SAMPLES_PER_PIXEL)
            }
        }
    }

    let elapsed_time = now.elapsed();
    println!("Time: {:.2} s ({} ms)",
             elapsed_time.as_secs_f32(), elapsed_time.as_millis());

    image.save("result.bmp")
        .expect("Failed to save image");
}
