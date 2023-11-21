//! MPI-based implementation of CommGroup.
use crate::{CommGroup, DataType};
use lazy_static::lazy_static;
use mpi_sys::{
    MPI_Comm, MPI_Comm_rank, MPI_Comm_size, MPI_Finalize, MPI_Get_count, MPI_Init_thread,
    MPI_Iprobe, MPI_Irecv, MPI_Isend, MPI_Request, MPI_Test, RSMPI_COMM_WORLD, RSMPI_STATUS_IGNORE,
    RSMPI_THREAD_MULTIPLE, RSMPI_UINT8_T,
};
use std::future::Future;
use std::mem::MaybeUninit;
use std::os::raw::c_int;
use std::pin::Pin;
use std::sync::Mutex;

lazy_static! {
    static ref MPI_INIT_LOCK: Mutex<i32> = Mutex::new(0);
}

pub fn init_standard_mpi() -> MPICommGroup {
    unsafe {
        let mut init_lock = MPI_INIT_LOCK.lock().unwrap();
        if *init_lock != 0 {
            panic!("Attempted to initialize multiple MPI instances");
        }
        *init_lock = 1;
        let mut provided: c_int = 0;
        MPI_Init_thread(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            RSMPI_THREAD_MULTIPLE,
            &mut provided,
        );
        assert_eq!(provided, RSMPI_THREAD_MULTIPLE);
        MPICommGroup {
            comm: RSMPI_COMM_WORLD,
        }
    }
}

impl Drop for MPICommGroup {
    fn drop(&mut self) {
        unsafe {
            MPI_Finalize();
            let mut init_lock = MPI_INIT_LOCK.lock().unwrap();
            *init_lock = 0;
        }
    }
}

pub struct MPICommGroup {
    comm: MPI_Comm,
}

// TODO: Could the internal MPI Communicator be modified during the
//       interleaving of an async call? This is potential UB.

impl CommGroup for MPICommGroup {
    fn rank(&self) -> u32 {
        unsafe {
            let mut value = 0;
            MPI_Comm_rank(self.comm, &mut value);
            value.try_into().unwrap()
        }
    }

    fn size(&self) -> u32 {
        unsafe {
            let mut value = 0;
            MPI_Comm_size(self.comm, &mut value);
            value.try_into().unwrap()
        }
    }

    fn send<T: DataType>(
        &self,
        data: &T,
        dest: u32,
        tag: u32,
    ) -> Pin<Box<dyn Future<Output = ()>>> {
        let buffer = bincode::serialize(data).unwrap();
        let comm = self.comm;
        Box::into_pin(Box::new(async move {
            internal_send(comm, &buffer, dest, tag).await
        }))
    }

    fn recv<T: DataType>(&self, source: u32, tag: u32) -> Pin<Box<dyn Future<Output = T>>> {
        let comm = self.comm;
        Box::into_pin(Box::new(async move {
            let buffer = internal_recv(comm, source, tag).await;
            bincode::deserialize(&buffer[..]).unwrap()
        }))
    }

    /// Broadcast data from this process to all other processes.
    ///
    /// bcast() and recv_bcast() implement a tree-based broadcast algorithm.
    fn bcast<T: DataType>(&self, data: &T) -> Pin<Box<dyn Future<Output = ()>>> {
        let buffer = bincode::serialize(data).unwrap();
        // TODO: Need to ensure that interleaved calls can't catch sends/recvs
        //       of broadcast
        let rank = self.rank();
        let size = self.size();
        let comm = self.comm;

        // Children ranks
        let a = 2 * rank + 1;
        let b = 2 * rank + 2;

        Box::into_pin(Box::new(async move {
            // Custom broadcast in order to allow for unknown size of receive
            // Send first to rank 0 if this is not rank 0
            if rank != 0 {
                internal_send(comm, &buffer, 0, 0).await;
            }
            // Then send to children
            if a < size {
                internal_send(comm, &buffer, a, 0).await;
            }
            if b < size {
                internal_send(comm, &buffer, b, 0).await;
            }
            ()
        }))
    }

    /// Receive a broadcast on all other processes.
    fn recv_bcast<T: DataType>(&self, source: u32) -> Pin<Box<dyn Future<Output = T>>> {
        let rank = self.rank();
        let size = self.size();
        let comm = self.comm;

        // Children ranks
        let a = 2 * rank + 1;
        let b = 2 * rank + 2;
        let parent = if rank == 0 {
            // Receive from the source if this is rank 0
            source
        } else {
            if rank % 2 == 0 {
                (rank - 2) / 2
            } else {
                (rank - 1) / 2
            }
        };

        Box::into_pin(Box::new(async move {
            // Receive value from parent rank
            let buffer = internal_recv(comm, parent, 0).await;

            // Send to children, if not source
            if a < size {
                internal_send(comm, &buffer, a, 0).await;
            }
            if b < size {
                internal_send(comm, &buffer, b, 0).await;
            }

            // Finally deserialize it
            bincode::deserialize(&buffer[..]).unwrap()
        }))
    }

    /// Scatter the data from this sending process to all processes (including
    /// this source process).
    ///
    /// data.len() must be divisible by the number of processes.
    fn scatter<T: DataType + 'static>(&self, data: &[T]) -> Pin<Box<dyn Future<Output = Vec<T>>>> {
        let comm = self.comm;
        let rank = self.rank();
        let size = self.size() as usize;

        assert_eq!(data.len() % size, 0);
        let subsize = data.len() / size;

        let data: Vec<T> = data.to_vec();
        Box::into_pin(Box::new(async move {
            for i in 0..size as u32 {
                if i == rank {
                    continue;
                }
                let idx = i as usize * subsize;
                let subdata = &data[idx..idx + subsize];
                let buffer = bincode::serialize(subdata).unwrap();
                internal_send(comm, &buffer, i, 0).await;
            }

            let rank = rank as usize;
            data[rank * subsize..(rank + 1) * subsize].to_vec()
        }))
    }

    /// Receive scattered data from a root process.
    fn scatter_recv<T: DataType>(&self, root: u32) -> Pin<Box<dyn Future<Output = Vec<T>>>> {
        let comm = self.comm;
        Box::into_pin(Box::new(async move {
            let buffer = internal_recv(comm, root, 0).await;
            bincode::deserialize(&buffer).unwrap()
        }))
    }

    /// Gather data from each process (including this one) and store in vector.
    fn gather<T: DataType + 'static>(&self, data: &[T]) -> Pin<Box<dyn Future<Output = Vec<T>>>> {
        let comm = self.comm;
        let rank = self.rank();
        let size = self.size();

        let mut data = Some(data.to_vec());
        Box::into_pin(Box::new(async move {
            let mut result = vec![];
            for i in 0..size {
                if i == rank {
                    if let Some(data) = data.take() {
                        result.extend(data);
                    }
                } else {
                    let data = internal_recv(comm, i, 0).await;
                    let subbuf: Vec<T> = bincode::deserialize(&data).unwrap();
                    result.extend(subbuf);
                }
            }
            result
        }))
    }

    /// Send data to a root process to be gathered.
    fn gather_send<T: DataType>(&self, root: u32, data: &[T]) -> Pin<Box<dyn Future<Output = ()>>> {
        let buffer = bincode::serialize(data).unwrap();
        let comm = self.comm;
        Box::into_pin(Box::new(async move {
            internal_send(comm, &buffer, root, 0).await;
        }))
    }
}

async fn internal_send(comm: MPI_Comm, buffer: &[u8], dest: u32, tag: u32) {
    unsafe {
        let mut req = MaybeUninit::uninit();
        MPI_Isend(
            buffer.as_ptr() as *const _,
            buffer.len().try_into().unwrap(),
            RSMPI_UINT8_T,
            dest.try_into().unwrap(),
            tag.try_into().unwrap(),
            comm,
            req.as_mut_ptr(),
        );
        let mut req = req.assume_init();

        while !test_request(&mut req).await {}
    }
}

async fn internal_recv(comm: MPI_Comm, source: u32, tag: u32) -> Vec<u8> {
    unsafe {
        let source: c_int = source.try_into().unwrap();
        let tag: c_int = tag.try_into().unwrap();

        let mut buffer = vec![];

        loop {
            if let Some(count) = probe_recv(comm, source, tag).await {
                buffer.resize(count, 0);
                break;
            }
        }

        let mut req = MaybeUninit::uninit();
        MPI_Irecv(
            buffer.as_mut_ptr() as *mut _,
            buffer.len().try_into().unwrap(),
            RSMPI_UINT8_T,
            source,
            tag,
            comm,
            req.as_mut_ptr(),
        );
        let mut req = req.assume_init();

        while !test_request(&mut req).await {}
        buffer
    }
}

/// Probe for a receive message.
async unsafe fn probe_recv(comm: MPI_Comm, source: c_int, tag: c_int) -> Option<usize> {
    let mut status = MaybeUninit::uninit();
    let mut flag = 0;
    MPI_Iprobe(source, tag, comm, &mut flag, status.as_mut_ptr());
    let status = status.assume_init();
    if flag != 0 {
        let mut count = 0;
        MPI_Get_count(&status, RSMPI_UINT8_T, &mut count);
        Some(count.try_into().unwrap())
    } else {
        None
    }
}

/// Test if the request is complete.
async unsafe fn test_request(req: &mut MPI_Request) -> bool {
    let mut flag = 0;
    MPI_Test(req, &mut flag, RSMPI_STATUS_IGNORE);
    flag != 0
}
