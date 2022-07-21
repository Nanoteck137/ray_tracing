use glam::f32::Vec3;

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

fn hit_sphere(center: Vec3, radius: f32, ray: &Ray) -> f32 {
    let oc = ray.origin - center;
    let a = ray.dir.length_squared();
    let half_b = oc.dot(ray.dir);
    let c = oc.length_squared() - radius * radius;
    let discriminant = half_b * half_b - a * c;

    if discriminant < 0.0 {
        return -1.0;
    } else {
        return (-half_b - discriminant.sqrt()) / a;
    }
}

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

fn main() {
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 1920;
    let image_height = (image_width as f32 / aspect_ratio) as usize;
    println!("Width: {} Height: {}", image_width, image_height);

    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;
    println!("Viewport Width: {} Height: {}", viewport_width, viewport_height);

    let origin = Vec3::new(0.0, 0.0, 0.0);
    let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
    let vertical = Vec3::new(0.0, viewport_height, 0.0);
    let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

    let mut image = bmp::Image::new(image_width as u32, image_height as u32);

    for y in 0..image_height {
        let per = ((y as f32 / image_height as f32) * 100.0) as u32;
        print!("\rWorking: {}%", per);
        for x in 0..image_width {
            let u = x as f32 / image_width as f32;
            let v = y as f32 / image_height as f32;

            let dir = lower_left_corner + horizontal * u + vertical * v  - origin;
            let ray = Ray::new(origin, dir);
            let color = ray_color(&ray);

            /*
            let r = x as f32 / image_width as f32;
            let g = y as f32 / image_height as f32;
            let b = 0.25;
            */

            let r = (color.x * 255.0) as u8;
            let g = (color.y * 255.0) as u8;
            let b = (color.z * 255.0) as u8;

            image.set_pixel(x as u32, (image_height - y - 1) as u32,
                            bmp::Pixel::new(r, g, b));
        }
    }
    println!();

    image.save("result.bmp")
        .expect("Failed to save image");
}
