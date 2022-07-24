#![allow(dead_code)]

use std::path::Path;
use glam::f32::{ Vec2, Vec3, Vec4, Mat4 };

const SAMPLES_PER_PIXEL: usize = 100;
const MAX_DEPTH: usize = 4;

mod cpu;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Debug)]
struct UniformData {
    view_matrix_inv: [f32; 4 * 4],
    projection_matrix_inv: [f32; 4 * 4],
}

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

        let dir = self.projection_matrix_inv * Vec4::from((uv, 0.0, 1.0));
        let dir = dir.truncate();

        let dir = self.view_matrix_inv * Vec4::from((dir, 0.0));
        let dir = dir.truncate();

        let dir = dir.normalize();

        Ray::new(origin, dir)
    }
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

#[derive(Clone)]
struct Material {
    color: Vec3,

    metallic: bool,
    metallic_strength: f32,

    dielectric: bool,
    ir: f32,
}

#[derive(Clone)]
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

#[derive(Clone)]
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

#[allow(dead_code)]
fn random_in_hemisphere(normal: Vec3) -> Vec3 {
    let unit_sphere = random_in_unit_sphere_vec3();
    if unit_sphere.dot(normal) > 0.0 {
        return unit_sphere;
    } else {
        return -unit_sphere;
    }
}

#[allow(dead_code)]
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
        ir: 1.4,
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
        metallic_strength: 1.0,

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

fn print_adapter(adapter: &wgpu::Adapter) {
    let info = adapter.get_info();

    let typ = match info.device_type {
        wgpu::DeviceType::Other => "Other",
        wgpu::DeviceType::IntegratedGpu => "Integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "Discrete GPU",
        wgpu::DeviceType::VirtualGpu => "Virtual GPU",
        wgpu::DeviceType::Cpu => "Software",
    };

    let backend = match info.backend {
        wgpu::Backend::Empty  => "Empty",
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal  => "Metal",
        wgpu::Backend::Dx12   => "Direct3D 12",
        wgpu::Backend::Dx11   => "Direct3D 11",
        wgpu::Backend::Gl     => "OpenGL",
        wgpu::Backend::BrowserWebGpu => "Browser GPU",
    };

    println!("{} ({}): {}", backend, typ, info.name);
}

#[allow(dead_code)]
struct GpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn initialize_wgpu() -> GpuContext {
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    println!("---- Available adapters ----");
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        print_adapter(&adapter);
    }
    println!("----------------------------");

    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }).await.unwrap();

    print!("Using: ");
    print_adapter(&adapter);

    let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
        features: wgpu::Features::empty(),
        limits: wgpu::Limits::default(),
        label: None,
    }, None).await.unwrap();

    GpuContext {
        instance,
        adapter,
        device,
        queue
    }
}

fn padded_bytes_per_row(width: usize) -> usize {
    let bytes_per_row = width * 4;
    let padding = (256 - bytes_per_row % 256) % 256;
    bytes_per_row + padding
}

async fn test_compute(context: &GpuContext,
                      shader_binary: &[u8],
                      camera: &Camera)
    -> Vec<Vec3>
{
    let device = &context.device;
    let queue = &context.queue;

    let image_width = 1920usize;
    let image_height = 1080usize;

    let output_texture_size = wgpu::Extent3d {
        width: image_width as u32,
        height: image_height as u32,
        depth_or_array_layers: 1,
    };

    let output_texture = device.create_texture(&wgpu::TextureDescriptor{
        label: Some("Compute Output Texture"),
        size: output_texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC |
            wgpu::TextureUsages::STORAGE_BINDING,
    });

    let padded_size = padded_bytes_per_row(image_width);
    let unpadded_size = image_width * 4;

    let staging_buffer_size = padded_size * image_height;
    println!("Staging Buffer Size: {}", staging_buffer_size);

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: staging_buffer_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    use wgpu::util::DeviceExt;

    let uniform_data = UniformData {
        view_matrix_inv: camera.view_matrix_inv.to_cols_array(),
        projection_matrix_inv: camera.projection_matrix_inv.to_cols_array(),
    };

    let uniform_buffer = device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniform_data]),
            usage: wgpu::BufferUsages::UNIFORM,
        }
    );

    let shader_desc = wgpu::ShaderModuleDescriptor {
        label: Some("test.comp Compute Shader"),
        source: wgpu::util::make_spirv(shader_binary),
    };

    let compute_module = device.create_shader_module(shader_desc);

    let compute_pipeline =
        device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &compute_module,
                entry_point: "main",
            }
    );

    let output_texture_view =
        output_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &output_texture_view,
                ),
            },

            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            }
        );

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_width = 8;
        let workgroup_height = 8;

        let dispatch_width = (image_width + workgroup_width - 1) / workgroup_width;
        let dispatch_height = (image_height + workgroup_height - 1) / workgroup_height;
        compute_pass.dispatch_workgroups(dispatch_width as u32,
                                         dispatch_height as u32,
                                         1);
    }

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            aspect: wgpu::TextureAspect::All,
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::ImageCopyBuffer {
            buffer: &staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(padded_size as u32),
                rows_per_image: std::num::NonZeroU32::new(image_height as u32),
            },
        },
        output_texture_size,
    );

    queue.submit([encoder.finish()]);

    let buffer_slice = staging_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, move |_v| {});

    device.poll(wgpu::Maintain::Wait);

    let padded_data = buffer_slice.get_mapped_range();

    let mut pixels: Vec<u8> = vec![0; unpadded_size * image_width as usize];
    for (padded, pixels) in padded_data
        .chunks_exact(padded_size)
        .zip(pixels.chunks_exact_mut(unpadded_size))
    {
        pixels.copy_from_slice(&padded[..unpadded_size]);
    }

    let mut new_pixels: Vec<Vec3> = vec![Vec3::new(0.0, 0.0, 0.0); image_width * image_height];
    for i in 0..(image_width * image_height) {
        let r = pixels[i * 4 + 0] as f32 / 255.0;
        let g = pixels[i * 4 + 1] as f32 / 255.0;
        let b = pixels[i * 4 + 2] as f32 / 255.0;
        let _a = pixels[i * 4 + 3] as f32 / 255.0;

        let pixel = Vec3::new(r, g, b);
        new_pixels[i] = pixel;
    }

    std::mem::drop(padded_data);
    staging_buffer.unmap();

    write_framebuffer_to_image("test.bmp",
                               image_width,
                               image_height,
                               &new_pixels);

    Vec::new()
}

fn hit_sphere(ray: &Ray, position: Vec3, radius: f32) -> bool {
    let oc = ray.origin - position;
    let a = ray.dir.dot(ray.dir);
    let half_b = oc.dot(ray.dir);
    let c = oc.dot(oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;

    return discriminant > 0.0;
}

fn shoot_ray(ray: &Ray) -> Vec3 {
    if hit_sphere(ray, Vec3::new(0.0, 0.0, -10.0), 1.0) {
        return Vec3::new(1.0, 0.0, 0.0);
    }

    let dir = ray.dir.normalize();
    let t = 0.5 * (dir.y + 1.0);

    let color1 = Vec3::new(1.0, 1.0, 1.0);
    let color2 = Vec3::new(0.5, 0.7, 1.0);

    return (1.0 - t) * color1 + t * color2;
}

fn main() {
    use pollster::FutureExt;

    env_logger::init();

    let aspect_ratio = 16.0 / 9.0;
    let image_width = 1920;
    let image_height = (image_width as f32 / aspect_ratio) as usize;
    println!("Width: {} Height: {}", image_width, image_height);

    println!("Available Cores: {}", num_cpus::get());

    let position = Vec3::new(0.0, 0.0, 0.0);
    let look_at = Vec3::new(0.0, 0.0, -1.0);
    let fov = 20.0;
    let camera = Camera::new(position, look_at, fov, aspect_ratio);

    let shader_binary = include_bytes!(env!("test.comp.spv"));

    let context = initialize_wgpu().block_on();
    let framebuffer = test_compute(&context, shader_binary, &camera).block_on();

    return;

    let world = create_random_world();
    // let world = create_simple_world();

    let tile_width = 64;
    let tile_height = 64;

    let job_queue = cpu::create_job_queue(image_width, image_height,
                                          tile_width, tile_height);


    let framebuffer =
        cpu::debug_write_job_queue_to_framebuffer(image_width,
                                                  image_height,
                                                  &job_queue);
    write_framebuffer_to_image("debug_tiles.bmp",
                               image_width,
                               image_height,
                               &framebuffer);

    {
        println!("-- Singlethreaded --");
        // NOTE(patrik): Singlethreaded Version
        let dispatch = cpu::Dispatch::new(job_queue.clone(),
                                          image_width, image_height,
                                          MAX_DEPTH, SAMPLES_PER_PIXEL,
                                          camera.clone(), world.clone());

        let framebuffer = cpu::dispatch_work_cpu_singlethreaded(dispatch);

        write_framebuffer_to_image("singlethreaded.bmp",
                                   image_width,
                                   image_height,
                                   &framebuffer);
    }

    {
        // NOTE(patrik): Multithreaded Version
        let num_threads = num_cpus::get();

        println!("-- Multithreaded ({}) --", num_threads);
        let dispatch = cpu::Dispatch::new(job_queue,
                                          image_width, image_height,
                                          MAX_DEPTH, SAMPLES_PER_PIXEL,
                                          camera, world);

        let framebuffer = cpu::dispatch_work_cpu_multithreaded(dispatch,
                                                               num_threads);

        write_framebuffer_to_image("multithreaded.bmp",
                                   image_width,
                                   image_height,
                                   &framebuffer);
    }

    // let framebuffer = dispatch_work_gpu_compute_shader();
}
