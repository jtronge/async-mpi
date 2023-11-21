# Async MPI Rust Implementation

Implementation of async MPI library. Ideas borrowed from several places
including:

* <https://github.com/rsmpi/rsmpi>
* <https://github.com/rufflewind/mpi_futures>

## Notes

This library takes a slightly different approach to writing distributed
parallel message passing code in MPI. For one, it operates on types which
can be serialized with serde, and most operations are asynchronous. This should
hopefully simplify message-passing operations somewhat.

RSMPI, the current Rust MPI bindings library, tries to get very close to the
original interface; many function wrappers are designed to call directly into
the underlying library with very minimal processing in Rust, which is made
possible with the use of a procedural macro that implements derived datatypes
around certain Rust types. For example, the `receive_into` function is able to
write directly into any buffer that implements the `Equivalence` trait. Now,
the design of RSMPI allows the bindings to have an extremely low overhead when
compared with direct C calls, but at the cost of introducing complexity, such
as in the immediate/non-blocking calls, where we need to avoid deallocating
borrowed data, as well as potentially undefined behavior due to the way
underlying MPI implementations match function calls and handle invalid
arguments with collective calls.

This library attempts to address some of the above issues through the use of
serialization and custom collectives, while also introducing some level of
overhead. There are many places where allocations and extra copies need to be
made that would never occur in RSMPI. Some of these could possibly be eliminated
with further optimization, given that the current implementation is rather naive,
but some additional allocations are necessary in order to maintain this less
complicated interface. This is not meant, by any means, to replace RSMPI or any
other MPI library, but more to be a prototype interface to MPI that could
prove useful to some applications.
