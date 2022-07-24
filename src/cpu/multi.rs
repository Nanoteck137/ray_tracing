use std::time::{ Instant, Duration };
use std::collections::VecDeque;
use std::sync::{ Arc, Mutex, RwLock };
use std::thread::JoinHandle;

use super::compose_final_framebuffer;
use super::execute_job;
use super::job::{ TileJob, JobData, JobResult };
use super::Dispatch;
use crate::{ World, Camera };
use glam::f32::Vec3;

type JobQueue = Arc<Mutex<VecDeque<TileJob>>>;
type JobResults = Arc<Mutex<Vec<JobResult>>>;

struct WorkerData {
    image_width: usize,
    image_height: usize,

    max_depth: usize,
    samples_per_pixel: usize,

    world: Arc<RwLock<World>>,
    camera: Camera,

    job_queue: JobQueue,
    job_results: JobResults,
}

fn worker(data: WorkerData) {
    loop {
        let job = {
            let mut lock = data.job_queue.lock().unwrap();
            lock.pop_front()
        };

        if let Some(job) = job {
            let world_lock = data.world.read().unwrap();

            let job_data = JobData {
                image_width: data.image_width,
                image_height: data.image_height,

                max_depth: data.max_depth,
                samples_per_pixel: data.samples_per_pixel,

                camera: &data.camera,
                world: &world_lock,
            };

            let result = execute_job(&job, &job_data);

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

fn dispatch_threads(num_threads: usize,
                    image_width: usize,
                    image_height: usize,
                    max_depth: usize,
                    samples_per_pixel: usize,
                    world: Arc<RwLock<World>>,
                    camera: &Camera,
                    job_queue: &JobQueue,
                    job_results: &JobResults)
    -> Vec<JoinHandle<()>>
{
    let mut thread_join_handles = Vec::with_capacity(num_threads);
    for _thread_id in 0..num_threads {
        let data = WorkerData {
            image_width,
            image_height,

            max_depth,
            samples_per_pixel,

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

    thread_join_handles
}

fn wait_for_threads(thread_join_handles: Vec<JoinHandle<()>>,
                    job_results: &JobResults,
                    num_jobs: usize)
{
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
}

pub fn dispatch_work_cpu_multithreaded(dispatch: Dispatch, num_threads: usize)
    -> Vec<Vec3>
{
    // dispatch threads
    // wait for threads to finish
    // compose the final framebuffer

    let num_jobs = dispatch.job_queue.len();

    let job_queue = Arc::new(Mutex::new(dispatch.job_queue));
    let job_results = Arc::new(Mutex::new(Vec::new()));

    let world = Arc::new(RwLock::new(dispatch.world));
    let camera = dispatch.camera;

    let now = Instant::now();

    let thread_handles =
        dispatch_threads(num_threads,
                         dispatch.image_width,
                         dispatch.image_height,
                         dispatch.max_depth,
                         dispatch.samples_per_pixel,
                         world,
                         &camera,
                         &job_queue,
                         &job_results);

    wait_for_threads(thread_handles,
                     &job_results,
                     num_jobs);

    let elapsed_time = now.elapsed();
    println!("Time: {:.2} s ({} ms)",
             elapsed_time.as_secs_f32(), elapsed_time.as_millis());

    let job_results_lock = job_results.lock().unwrap();
    compose_final_framebuffer(dispatch.image_width,
                              dispatch.image_height,
                              dispatch.samples_per_pixel,
                              &job_results_lock)
}
