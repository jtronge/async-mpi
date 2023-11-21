use async_mpi::{init_standard_mpi, CommGroup};
use futures::executor;

fn main() {
    let cg = init_standard_mpi();
    let rank = cg.rank();
    let size = cg.size();

    executor::block_on(async {
        if rank == 0 {
            let buffer: Vec<f64> = (0..size * 8).map(|i| i as f64).collect();
            let mut data = cg.scatter(&buffer).await;
            for elm in &mut data {
                *elm *= 4.0;
            }
            let result = cg.gather(&data).await;
            println!("Gather result of {:?}", result);
        } else {
            let mut data: Vec<f64> = cg.scatter_recv(0).await;
            for elm in &mut data {
                *elm *= 4.0;
            }
            cg.gather_send(0, &data).await;
        }
    })
}
