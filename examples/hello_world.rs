use async_mpi::{init_standard_mpi, CommGroup};

fn main() {
    let cg = init_standard_mpi();
    println!("Hello world from rank {} of {}", cg.rank(), cg.size());
}
