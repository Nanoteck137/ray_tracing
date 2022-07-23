use std::time::Instant;
use std::collections::VecDeque;

use super::compose_final_framebuffer;
use super::execute_job;
use super::job::{ TileJob, JobData };
use crate::{ World, Camera };
use glam::f32::Vec3;

pub fn dispatch_work_cpu_singlethreaded(mut job_queue: VecDeque<TileJob>,
                                        image_width: usize,
                                        image_height: usize,
                                        max_depth: usize,
                                        samples_per_pixel: usize,
                                        camera: Camera,
                                        world: World)
    -> Vec<Vec3>
{
    let now = Instant::now();

    let mut job_results = Vec::new();
    while !job_queue.is_empty() {
        let job = job_queue.pop_front().unwrap();

        let data = JobData {
            image_width,
            image_height,

            max_depth,
            samples_per_pixel,

            camera: &camera,
            world: &world,
        };

        let result = execute_job(&job, &data);
        job_results.push(result);
    }

    let elapsed_time = now.elapsed();
    println!("Time: {:.2} s ({} ms)",
             elapsed_time.as_secs_f32(), elapsed_time.as_millis());

    compose_final_framebuffer(image_width,
                              image_height,
                              samples_per_pixel,
                              &job_results)
}
