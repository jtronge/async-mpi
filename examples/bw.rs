//! Bandwidth example (partially based on the version in the OSU Micro
//! Benchmarks).
use async_mpi::{init_standard_mpi, CommGroup};
use futures::{executor, future};
use std::time::Instant;

fn main() {
    let cg = init_standard_mpi();
    let rank = cg.rank();
    let size = cg.size();

    if size != 2 {
        panic!("The bandwidth test requires exactly two processes");
    }

    executor::block_on(async {
        if rank == 0 {
            println!("{:<8}\t{:>8}", "size", "MiB/s");
        }

        let mut buffer = Vec::<u8>::new();
        let mut size = 1;
        while size <= 65536 {
            let mut bw = 0.0;

            buffer.resize(size, 0);
            if rank == 0 {
                // Warmup
                future::join_all((0..16).map(|_| cg.send(&buffer, 1, 0))).await;

                let start = Instant::now();
                future::join_all((0..1024).map(|_| cg.send(&buffer, 1, 0))).await;
                let end = Instant::now();
                let total_time = end.duration_since(start).as_secs_f64();
                let len = buffer.len() as f64;
                bw = 1024.0 * len / (total_time * 1024.0 * 1024.0);
            } else {
                // Warmup
                future::join_all((0..16).map(|_| cg.recv::<Vec<u8>>(0, 0))).await;

                future::join_all((0..1024).map(|_| cg.recv::<Vec<u8>>(0, 0))).await;
            }

            if rank == 0 {
                println!("{:<8}\t{:>8.2}", size, bw);
            }
            size *= 2;
        }
    });
}
