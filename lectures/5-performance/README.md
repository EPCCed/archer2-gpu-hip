template: titleslide
# Performance



---
# Performance?

We should now have a functioning, if simple, GPU program which moves
some data to and from the device and executes a kernel.

What influences the performance of such a program?



---
# Parallelism

Amdahl's law states that the parallel performance of a program is
limited by the fraction of code that is serial.

In the GPU context, this gives rise to two concerns.

1. Kernel code is potentially parallel.
2. Host code is definitely serial (including host-device transfers).

This may mean that additional work is required to expose parallelism,
or eliminate host operations in favour of device operations.



---
# Occupancy

- The GPU has many CU cores that can potentially be used. Having very
many blocks of work available at any one time is said to favour
high *occupancy*.

- High occupancy is achieved by codes that have a high degree of thread
parallelism. Typically, a GPU kernel wants at least 10<sup>5</sup> threads
to be effective, i.e. the problem <br>space should have this number of elements.

- There may be little benefit in using a GPU if the problem as coded does not
have this degree of (data) parallelism.



---
# Two dimensional example

Consider a two-dimensional loop.

```cpp
   int NX = 512;
   int NY = 512;
   ...
   for (int i = 0; i < NX; i++) {
     for (int j = 0; j < NY; j++) {
       /* ... work for element (i, j) ... */
     }
   }
```

If we parallelised the inner loop only, we would have work for at most
512 threads, <br>i.e. two blocks of 256 threads per block.

This would clearly be poor occupancy.

If we parallelised both loops, we would have 512 × 512 = 262,144 threads
(1024 blocks), which is much better as many more CU cores will now be utilised.



---
# Memory usage

On GPUs, many cores need to be occupied (i.e. given work) at the same time to avoid
under-exploiting the device's resources.

This means the optimum memory access patterns for CPU and GPU are not the same.



---
# CPU: caching behaviour

A thread in a CPU code favours consecutive memory accesses, e.g. in C, recall that it
is the right-most index that runs fastest in memory.

```cpp
   for (int i = 0; i < NX; i++) {
     for (int j = 0; j < NY; j++) {
       a[i][j] = 0.0;
     }
   }
```

Such an order yields favourable cache behaviour: the memory accesses made by a single thread are contiguous,
optimising performance.



---
# GPU: coalescing behaviour

The situation is different for GPU global memory: we need to have waveforms of consecutive threads load consecutive memory
locations in a contiguous block.

Consider a one-dimensional example.

```cpp
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  a_output[i] = a_input[i];
```

Here, there is no issue, consecutive threads (those with consecutive `x` index) access consecutive memory locations.



---
# GPU: coalescing in 2D

The code below initialises `i` and iterates over `NY` using `j`.

```cpp
  int i = blockIdx.x*blockDim.x + threadIdx.x;

  for (int j = 0; j < NY; j++) {
    a_output[i][j] = a_input[i][j];
  }
```

Hence, each thread makes `NY` consecutive accesses to the arrays.<br>
This does **not** favour coalesced access.



---
# GPU: coalescing in 2D

<span style="color:gray">The code below initialises `i` and iterates over `NY` using `j`.</span>

```
  ...
```

<span style="color:gray">Hence, each thread makes `NY` consecutive accesses to the arrays.<br></span>
<span style="color:gray">This does **not** favour coalesced access.</span>

We want consecutive threads to have consecutive accesses, i. e. initialise `j` and <br>
iterate over `NX` using `i`.

```cpp
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  for (int i = 0; i < NX; i++) {
    a_output[i][j] = a_input[i][j];
  }
```

This is the favoured pattern for GPU: coalescing favours a given thread having <br>*strided* memory access.



---
# <span style="color:red">Exercise:</span> Coalescing (1/5)

Using the code template [exercise_dger.hip.cpp](../../exercises/5-performance/1-coalescing/), implement
a kernel that computes the following matrix operation.

A<sub>ij</sub> = A<sub>ij</sub> + α x<sub>i</sub> y<sub>j</sub> ,

where `A` is an `m` × `n` matrix, `x` is a vector of length `m`, `y` is a vector of length `n` and α is a constant. All terms are of type `double`.

Adopt a flattened one-dimensional indexing for matrix `A`: element row `i` and column `j` is addressed as `A[i*ncol + j]`.

<br>
As this is also a performance issue, `rocprof` is called from within the `srun` command, see [submission script](../../exercises/5-performance/1-coalescing/).
The `rocprof` utility will profile the GPU kernel, outputting timing info at the end of the execution. Try to keep a note of the time taken by the
kernel at each stage (reported in nanoseconds (`ns`) by `rocprof`).

<br>
*A suggested procedure is outlined on the next slides.* 



---
# <span style="color:red">Exercise:</span> Coalescing (2/5)


Check the template to see that the matrix and vectors have been established
in device memory. Note that the template uses a particular HIP API call for
initialising the matrix elements ro zero.

```cpp
hipError_t hipMemset(void * dptr, int value, size_t sizeBytes);
```

The template should compile and run, but will not compute the correct answer as the
supplied kernel stub does nothing.



---
# <span style="color:red">Exercise:</span> Coalescing (3/5)

Implement the most simple kernel in which the update is entirely serialised.

```cpp
int tid = blockIdx.x*blockDim.x + threadIdx.x;

if (tid == 0) {
  for (int i = 0; i < mrow; i++) {
    for (int j = 0; j < ncol; j++) {
       a[ncol*i + j] = a[ncol*i + j] + alpha*x[i]*y[j];
     }
  }
}
```

Check the execution configuration and run the code to ensure it reports the correct answer.



---
# <span style="color:red">Exercise:</span> Coalescing (4/5)

Eliminate the `i`-loop and re-check the kernel launch parameters to provide parallelism over the rows of the matrix.
Remember to allow that the problem size is not a whole number of blocks.

In addition, eliminate the `j`-loop to have parallelism over both rows and columns. You will need to introduce two dimensions
in the abstract description.

```cpp
int j = blockIdx.y*blockDim.y + threadIdx.y;
```

Then make an appropriate adjustment to the kernel launch parameters.<br>
**Hint**: *keep the same total number of threads per block, but make the block
must become 2D*.



---
# <span style="color:red">Exercise:</span> Coalescing (5/5)

Lastly, is your resultant code getting the coalescing right?

<br>
Consecutive threads, that is, threads with consecutive `x`-index, should access consecutive memory locations.



---
# <span style="color:green">Finished?</span>

- If we had not used `hipMemset()` to initialise the device values for the matrix, what other options to initialise these values on the device
are available to us?

  - `hipMemset()` is limited in that it can **not** be used to initialise using <br>to non-zero values.

- For your best effort for the kernel, what is the overhead of the actual kernel launch (`hipLaunchKernel` in the profile) compared with the time taken for the
kernel itself?

  - These can be found in `results.stats.csv` and `results.hip_stats.csv`, or in `results.json`.

- What's the overhead for the host-device transfers?



---
# <span style="color:red">Next Lecture</span>

<br>
## [Managed Memory](../6-managed-memory)