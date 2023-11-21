//! Ring example (partially based on Open MPI's ring_c example).
use async_mpi::{init_standard_mpi, CommGroup};
use futures::executor;

fn main() {
    let cg = init_standard_mpi();
    let rank = cg.rank();
    let size = cg.size();
    let next = (rank + 1) % size;
    let prev = (rank + size - 1) % size;

    // MPI calls must be within an aysnc block
    executor::block_on(async {
        if rank == 0 {
            println!("Sending message on rank 0");
            let message: i32 = 10;
            cg.send(&message, next, 0).await;
        }

        // Send the integer message around in a loop and decrement it each time
        // it passes rank 0. Each rank quits once it sees a message of 0.
        loop {
            let mut message: i32 = cg.recv(prev, 0).await;
            if rank == 0 {
                println!("Decrementing message on rank 0 (current: {})", message);
                message -= 1;
            }
            cg.send(&message, next, 0).await;
            if message == 0 {
                break;
            }
        }

        // Receive last message to rank 0
        if rank == 0 {
            let _: i32 = cg.recv(prev, 0).await;
        }
    });
}
