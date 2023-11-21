use serde::{de::DeserializeOwned, Serialize};
use std::future::Future;
use std::pin::Pin;

pub trait DataType: Serialize + DeserializeOwned + Default + Clone {}

impl<T> DataType for T where T: Serialize + DeserializeOwned + Default + Clone {}

pub trait CommGroup {
    /// Get the rank of the process in the group.
    fn rank(&self) -> u32;
    /// Get the size of this communication group.
    fn size(&self) -> u32;
    /// Send data to a destination process.
    fn send<T: DataType>(&self, data: &T, dest: u32, tag: u32)
        -> Pin<Box<dyn Future<Output = ()>>>;
    /// Receive some data from a source process.
    fn recv<T: DataType>(&self, source: u32, tag: u32) -> Pin<Box<dyn Future<Output = T>>>;
    /// Broadcast data from this process to all other processes.
    fn bcast<T: DataType>(&self, data: &T) -> Pin<Box<dyn Future<Output = ()>>>;
    /// Receive a broadcast on all other processes.
    fn recv_bcast<T: DataType>(&self, source: u32) -> Pin<Box<dyn Future<Output = T>>>;
    /// Scatter the data from this sending process to all processes (including
    /// this source process).
    ///
    /// data.len() must be divisible by the number of processes.
    fn scatter<T: DataType + 'static>(&self, data: &[T]) -> Pin<Box<dyn Future<Output = Vec<T>>>>;
    /// Receive scattered data from a root process.
    fn scatter_recv<T: DataType>(&self, root: u32) -> Pin<Box<dyn Future<Output = Vec<T>>>>;
    /// Gather data from each process (including this one) and store in vector.
    fn gather<T: DataType + 'static>(&self, data: &[T]) -> Pin<Box<dyn Future<Output = Vec<T>>>>;
    /// Send data to a root process to be gathered.
    fn gather_send<T: DataType>(&self, root: u32, data: &[T]) -> Pin<Box<dyn Future<Output = ()>>>;
}

mod mpi;
pub use mpi::init_standard_mpi;
