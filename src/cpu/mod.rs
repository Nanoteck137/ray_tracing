use std::time::{ Instant, Duration };
use std::collections::VecDeque;
use std::sync::{ Arc, Mutex, RwLock };
use std::thread::JoinHandle;

use glam::f32::{ Vec3, Vec2 };

use crate::{ World, Camera };

pub struct TileJob {
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

struct JobData<'a> {
    image_width: usize,
    image_height: usize,

    max_depth: usize,
    samples_per_pixel: usize,

    camera: &'a Camera,
    world: &'a World,
}

fn execute_job(job: &TileJob, data: &JobData) -> JobResult {
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

type JobQueue = Arc<Mutex<VecDeque<TileJob>>>;
type JobResults = Arc<Mutex<Vec<JobResult>>>;

struct WorkerData {
    thread_id: usize,

    image_width: usize,
    image_height: usize,

    max_depth: usize,
    samples_per_pixel: usize,

    world: Arc<RwLock<World>>,
    camera: Camera,

    job_queue: JobQueue,
    job_results: JobResults,
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

fn dispatch_threads(image_width: usize,
                    image_height: usize,
                    max_depth: usize,
                    samples_per_pixel: usize,
                    world: Arc<RwLock<World>>,
                    camera: &Camera,
                    job_queue: &JobQueue,
                    job_results: &JobResults)
    -> Vec<JoinHandle<()>>
{
    let num_threads = 8;
    let mut thread_join_handles = Vec::with_capacity(num_threads);
    for thread_id in 0..num_threads {
        let data = WorkerData {
            thread_id,

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

pub fn dispatch_work_cpu_multithreaded(job_queue: VecDeque<TileJob>,
                                       image_width: usize,
                                       image_height: usize,
                                       max_depth: usize,
                                       samples_per_pixel: usize,
                                       camera: Camera,
                                       world: World)
    -> Vec<Vec3>
{
    // dispatch threads
    // wait for threads to finish
    // compose the final framebuffer

    let num_jobs = job_queue.len();

    let job_queue = Arc::new(Mutex::new(job_queue));
    let job_results = Arc::new(Mutex::new(Vec::new()));

    let world = Arc::new(RwLock::new(world));

    let now = Instant::now();

    let thread_handles =
        dispatch_threads(image_width,
                         image_height,
                         max_depth,
                         samples_per_pixel,
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
    compose_final_framebuffer(image_width,
                              image_height,
                              samples_per_pixel,
                              &job_results_lock)
}
