use std::time::Instant;

use glam::f32::Vec3;

use super::compose_final_framebuffer;
use super::execute_job;
use super::Dispatch;
use super::job::JobData;

pub fn dispatch_work_cpu_singlethreaded(dispatch: Dispatch) -> Vec<Vec3> {
    let now = Instant::now();

    let mut job_queue = dispatch.job_queue;

    let mut job_results = Vec::new();
    while !job_queue.is_empty() {
        let job = job_queue.pop_front().unwrap();

        let data = JobData {
            image_width: dispatch.image_width,
            image_height: dispatch.image_height,

            max_depth: dispatch.max_depth,
            samples_per_pixel: dispatch.samples_per_pixel,

            camera: &dispatch.camera,
            world: &dispatch.world,
        };

        let result = execute_job(&job, &data);
        job_results.push(result);
    }

    let elapsed_time = now.elapsed();
    println!("Time: {:.2} s ({} ms)",
             elapsed_time.as_secs_f32(), elapsed_time.as_millis());

    compose_final_framebuffer(dispatch.image_width,
                              dispatch.image_height,
                              dispatch.samples_per_pixel,
                              &job_results)
}
