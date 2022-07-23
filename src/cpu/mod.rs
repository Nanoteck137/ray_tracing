use std::time::{ Instant, Duration };
use std::collections::VecDeque;
use std::sync::{ Arc, Mutex, RwLock };

use glam::f32::{ Vec3, Vec2 };

use crate::{ World, Camera };

mod job;
mod single;
mod multi;

pub use job::{ create_job_queue, debug_write_job_queue_to_framebuffer };

use job::{ TileJob, JobData, JobResult };
use job::execute_job;

pub use single::dispatch_work_cpu_singlethreaded;
pub use multi::dispatch_work_cpu_multithreaded;

pub struct Dispatch {
    job_queue: VecDeque<TileJob>,

    image_width: usize,
    image_height: usize,

    max_depth: usize,
    samples_per_pixel: usize,

    camera: Camera,
    world: World,
}

impl Dispatch {
    pub fn new(job_queue: VecDeque<TileJob>,
               image_width: usize, image_height: usize,
               max_depth: usize, samples_per_pixel: usize,
               camera: Camera, world: World)
        -> Self
    {
        Self {
            job_queue,
            image_width,
            image_height,
            max_depth,
            samples_per_pixel,
            camera,
            world
        }
    }
}

fn compose_final_framebuffer(image_width: usize,
                             image_height: usize,
                             samples_per_pixel: usize,
                             job_results: &Vec<JobResult>)
    -> Vec<Vec3>
{
    let mut framebuffer =
        vec![Vec3::new(0.0, 0.0, 0.0); image_width * image_height];

    for result in job_results {
        for yoff in 0..result.height {
            for xoff in 0..result.width {
                let x = result.x + xoff;
                let y = result.y + yoff;
                let color = result.pixel_colors[xoff + yoff * result.width];

                let scale = 1.0 / samples_per_pixel as f32;

                let r = (color.x * scale).sqrt();
                let g = (color.y * scale).sqrt();
                let b = (color.z * scale).sqrt();

                let r = r.clamp(0.0, 0.999);
                let g = g.clamp(0.0, 0.999);
                let b = b.clamp(0.0, 0.999);

                framebuffer[x + y * image_width] = Vec3::new(r, g, b);
            }
        }
    }

    framebuffer
}
