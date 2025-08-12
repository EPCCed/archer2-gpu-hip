template: titleslide
# Streams



---
# Streams

- Streams are independent flows of sequential operations executed on the GPU.

- When you execute a kernel, or call `hipMemcpy()`, the necessary operations are submitted to the *default* stream
associated with the current GPU context.

- Operations submitted to the same stream serialise, i.e. they are executed in the order in which they are added to the stream.



---
# Streams

- <span style="color:gray">Streams are independent flows of sequential operations executed on the GPU.</span>

- <span style="color:gray">When you execute a kernel, or call `hipMemcpy()`, the necessary operations are submitted to the *default* stream
associated with the current GPU context.</span>

- <span style="color:gray">Operations submitted to the same stream serialise, i.e. they are executed in the order in which they are added to the stream.</span>

- By using multiple streams, you can overlap operations such as kernel execution and memory transfers, leading to better utilisation of the GPU.
  - We can also use streams to run more than one kernel simultaneously.



---
# Why streams

- **Overlapping Data Transfers and Computation**:
While one stream handles data transfer from the host to the device, another stream can execute a kernel. This overlap reduces idle time and maximizes GPU utilisation.

- **Concurrent Kernel Execution**:
Running multiple independent kernels simultaneously can speed up tasks like image processing, where different filters can be applied in parallel.

- **Asynchronous Memory Operations**:
Performing memory copies and kernel executions concurrently. For instance, copying data for the next computation while the current computation is still running.

- **Pipeline Processing**:
In video processing, one stream can decode frames while another processes the decoded frames, and a third stream encodes the processed frames.



---
# Why streams (continued)

- **Load Balancing**:
Distributing work across multiple streams to balance the load and avoid bottlenecks, especially in complex simulations or scientific computations.

- **Real-Time Data Processing**:
In real-time applications like autonomous driving, one stream can handle sensor data acquisition while another processes the data for decision-making.

- **Multi-GPU Systems**:
In systems with multiple GPUs, streams can be used to manage tasks across different GPUs, ensuring efficient resource utilisation.



---
# Stream management

A stream object is declared like so.

```cpp
hipStream_t stream;
```

The stream object must be initialised before it is used.

```cpp
hipStreamCreate(&stream);
```

It is released when it is no longer required.

```cpp
hipStreamDestroy(stream);
```

One can create an arbitrary number of streams.



---
# Asynchronous copies

Streams are used for the asynchronous version of `hipMemcpy()`.

```cpp
hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sz,
                          hipMemcpyKind kind, hipStream_t stream);
```

This operation uses the default stream if the final stream argument is omitted.



---
# Asynchronous copies

Streams are used for the asynchronous version of `hipMemcpy()`.

```cpp
hipError_t hipMemcpyAsync(void *dst, const void *src, size_t sz,
                          hipMemcpyKind kind, hipStream_t stream);
```

This operation uses the default stream if the final stream argument is omitted.

![:thumb](`hipMemcpyAsync` still uses both host and device memory references like `hipMemcpy` and unlike `hipMemPrefetchAsync`.)



---
# Synchronisation

To know when an asynchronous stream operation can be considered complete, and therefore
safe to make use of the result, we need to synchronise.

```cpp
hipStreamSynchronize(stream);
```

This routine will block until all pending operations in the stream have completed.



---
# Kernels

Kernels may also be submitted to a non-default stream by using an optional argument to the execution configuration.

```cpp
<<<dim3 blocks, dim3 threadsPerBlock, size_t shared, hipStream_t stream>>>
```

This matches the full form of the analogous `hipLaunchKernelGGL()`:

```cpp
hipError_t hipLaunchKernelGGL(const void *func, dim3 gridDim, dim3 blockDim,
                              void **args, size_t shared, hipStream_t stream);
```



---
# Pinned memory

Pinned (or page-locked) memory is a type of memory whose location is locked: it cannot be moved within physical memory nor swapped out to disk by the operating system.

Thus, the data held in such memory can be transferred to and from GPU more efficiently.

HIP can allocate and release pinned memory as shown below.

```cpp
double *h_ptr = NULL;

hipHostMalloc(&h_ptr, ndata*sizeof(double));

...

hipHostFree(h_ptr);
```

Such allocations are often used in conjunction with streams where efficiency is of paramount concern.



---
# Pinned memory benefits

- **Faster Data Transfers**: Pinned memory allows for faster data transfers between the host (CPU) and the device (GPU) because it avoids the overhead of paging.

- **Predictable Performance**: Since the memory is always resident in RAM, access times are more predictable, which is crucial for real-time applications.

- **Direct Access by GPU**: The GPU can directly access pinned memory without involving the CPU, enabling more efficient data transfers.



---
# Pinned memory benefits

- **Faster Data Transfers**: Pinned memory allows for faster data transfers between the host (CPU) and the device (GPU) because it avoids the overhead of paging.

- **Predictable Performance**: Since the memory is always resident in RAM, access times are more predictable, which is crucial for real-time applications.

- **Direct Access by GPU**: The GPU can directly access pinned memory without involving the CPU, enabling more efficient data transfers.

![:thumb](However, pinned memory is more limited than pageable memory, so it should be used judiciously.)



---
# <span style="color:red">Exercise:</span> `dger` operation (1/2)

We revisit a previous problem that implemented the `dger()` BLAS call.
For this exercise, a new working template is provided, [exercise_dger.hip.cpp](../../exercises/10-streams/1-dger).

Here, we illustrate the use of streams and page-locked host memory.

A suggested procedure is given on the next slide.



---
# <span style="color:red">Exercise:</span> `dger` operation (2/2)

1. For vectors "`x`" and "`y`" replace the relevant `hipMemcpy()` with
   an asynchronous operation using two different streams. Make sure
   that the data has reached the device before the kernel launch.

   While it is unlikely that this will have any significant beneficial effect in
   performance, it should be possible to view the result from `rocprof` and see
   the different streams in operation using a tool like [Perfetto](https://ui.perfetto.dev/).

2. Check you can replace the host allocations of "`x`" and "`y`" with
   `hipHostMalloc()` and make the appropriate adjustment to free
   resources at the end of execution.



---
# <span style="color:red">Next Lecture</span>

<br>
## [Graphs](../11-graphs)