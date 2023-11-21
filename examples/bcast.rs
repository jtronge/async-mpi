use async_mpi::{init_standard_mpi, CommGroup};
use futures::executor;
use std::time::SystemTime;

fn main() {
    let cg = init_standard_mpi();
    let rank = cg.rank();
    let size = cg.size();

    let data: Vec<i32> = executor::block_on(async {
        // Choose a pseudo random source for the second bcast
        let source = if rank == 0 {
            let dur = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap();
            let source = (dur.as_secs() as u32) % size;
            cg.bcast(&source).await;
            source
        } else {
            cg.recv_bcast(0).await
        };

        if rank == 0 {
            println!("Broadcasting from rank {}", source);
        }

        // Broadcast from that source
        if rank == source {
            let data = vec![1, 2, 3];
            cg.bcast(&data).await;
            data.clone()
        } else {
            cg.recv_bcast(source).await
        }
    });

    println!("Rank {} received broadcast of {:?}", rank, data);
}
