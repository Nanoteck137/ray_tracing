use std::collections::VecDeque;
use glam::f32::{ Vec3, Vec2 };

use crate::{ World, Camera };

pub struct TileJob {
    pub x: usize,
    pub y: usize,

    pub width: usize,
    pub height: usize,
}

pub struct JobResult {
    pub x: usize,
    pub y: usize,

    pub width: usize,
    pub height: usize,

    pub pixel_colors: Vec<Vec3>,
}

pub struct JobData<'a> {
    pub image_width: usize,
    pub image_height: usize,

    pub max_depth: usize,
    pub samples_per_pixel: usize,

    pub camera: &'a Camera,
    pub world: &'a World,
}

pub fn execute_job(job: &TileJob, data: &JobData) -> JobResult {
    let size = job.width * job.height;
    let mut result = Vec::with_capacity(size);

    let image_width = data.image_width;
    let image_height = data.image_height;

    for yoff in 0..job.height {
        for xoff in 0..job.width {
            let x = job.x + xoff;
            let y = job.y + yoff;

            let mut pixel_color = Vec3::new(0.0, 0.0, 0.0);
            for _sample in 0..data.samples_per_pixel {
                let u = (x as f32 + rand::random::<f32>()) / image_width as f32;
                let v = (y as f32 + rand::random::<f32>()) / image_height as f32;
                let u = u * 2.0 - 1.0;
                let v = v * 2.0 - 1.0;
                let uv = Vec2::new(u, v);

                let ray = data.camera.get_ray(uv);
                let color = data.world.shoot_ray(&ray, data.max_depth);
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

pub fn create_job_queue(image_width: usize,
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

pub fn debug_write_job_queue_to_framebuffer(image_width: usize,
                                            image_height: usize,
                                            job_queue: &VecDeque<TileJob>)
    -> Vec<Vec3>
{
    let mut result = vec![Vec3::new(0.0, 0.0, 0.0); image_width * image_height];
    for job in job_queue {
        for yoff in 0..job.height {
            for xoff in 0..job.width {
                let pixel_x = job.x + xoff;
                let pixel_y = job.y + yoff;

                let r = xoff as f32 / job.width as f32;
                let g = yoff as f32 / job.height as f32;
                let b = 0.0;

                result[pixel_x + pixel_y * image_width] = Vec3::new(r, g, b);
            }
        }
    }

    result
}
