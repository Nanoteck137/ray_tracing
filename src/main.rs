use math::Vec3;

mod math;

fn main() {
    let image_width = 256usize;
    let image_height = 256usize;

    let test = Vec3::new(1.0, 2.0, 3.0);
    let test2 = Vec3::new(3.0, 2.0, 1.0);
    println!("{}", test);
    println!("{}", test.length());
    println!("{}", test + test2);
    println!("{}", test - test2);
    println!("{}", test * test2);
    println!("{}", test / test2);
    println!("{}", -test);
    println!("{}", test + 2.0);
    println!("{}", test - 2.0);
    println!("{}", test * 2.0);
    println!("{}", test / 2.0);
    println!("{}", test.normalize());

    let mut image = bmp::Image::new(image_width as u32, image_height as u32);

    for y in 0..image_height {
        print!("\rWorking: {}%", ((y as f32 / image_height as f32) * 100.0) as u32);
        for x in 0..image_width {
            let r = x as f32 / image_width as f32;
            let g = y as f32 / image_height as f32;
            let b = 0.25;

            let r = (r * 255.0) as u8;
            let g = (g * 255.0) as u8;
            let b = (b * 255.0) as u8;

            image.set_pixel(x as u32, (image_height - y - 1) as u32,
                            bmp::Pixel::new(r, g, b));
        }
    }
    println!();

    image.save("result.bmp")
        .expect("Failed to save image");
}
