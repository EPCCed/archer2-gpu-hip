template: titleslide
# Shared Memory


---
# Shared memory

So far we have seen **global** memory, which is allocated by some
mechanism on the host, and is available in the kernel.

We have also used local variables in the kernel, which appear
on the stack in the expected fashion. Local variables are expected
to be held in registers and take on a distinct value for each thread, e.g.,
```cpp
  int i = blockDim.x*blockIdx.x + threadIdx.x;
```

While global memory is shared between all threads, the usage of
'shared memory' is reserved for something more specific in the
GPU context. This is discussed below.

---
# Independent accesses to global memory

In what we have seen so far, kernels have been used to replace
loops with independent iterations. For instance,
```cpp
  for (int i = 0; i < ndata; i++) {
    data[i] = 2.0*data[i];
  }
```
is replaced by a kernel with the body
```cpp
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  data[i] = 2.0*data[i];
```
As each thread accesses an independent location in (global)
memory, there are no potential conflicts.


---
# A different pattern

Consider a loop with the following pattern
```cpp
  double sum = 0.0;
  for (int i = 0; i < ndata; i++) {
    sum += data[i];
  }
```
The iterations are now coupled in some sense: all must accumulate
a value to the single memory location `sum`.

What would happen if we tried to run a kernel of the following form?
```cpp
  __global__ myKernel(int ndata, double *data, double *sum) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < ndata) *sum += data[i];
  }
```


---
# The problem

The problem lies in the fact that the increment is actually a
number of different operations which occur in order.
```cpp
   *sum += ndata[i];
```
1. Read the current value of `sum` from memory into a register;
2. Undertake the appropriate arithmetic in register;
3. Store the new result back to global memory.

If many threads are performing these operations in an uncontrolled
fashion, unexpected results can arise.

Such non-deterministic results are frequently referred to as race
conditions.


---
# The solution

In practice, potentially unsafe updates to any form of shared memory
must be protected by appropriate synchronisation: guarantees that
operations happen in the correct order.

For global memory, we require a so-called *atomic* update. For our
example above:
```cpp
  *sum += data[i];            /* WRONG: unsafe update */
  atomicAdd(sum, data[i]);    /* Correct: atomic update */
```
Such updates are usually implemented by some form of lock.

So the atomic update is a single unified operation on a single thread:
1. Obtain a lock on the relevant memory location (`sum`);
2. Read the existing value into register and update;
3. Store the result back to the global memory location;
4. Release the lock on that location.


---
# Note

`atomicAdd()` is an overloaded device function:
```cpp
__device__ int atomicAdd(int *address, int value);
__device__ double atomicAdd(double *address, double value);
```
and so on. The old value of the target variable is returned.


---
# Shared memory in blocks

There is an additional type of shared memory available in kernels
introduced using the `__shared__` memory space qualifier. E.g.,
```cpp
  __shared__ double tmp[THREADS_PER_BLOCK];
```
Shared memory is a type of memory that is accessible by all threads within the
same block in a GPU. It is faster than global memory because it is located
on-chip, close to the processing cores. This makes it ideal for storing data
that needs to be frequently accessed by multiple threads.

Common use cases:
1. **Matrix Multiplication**: Shared memory can be used to store sub-matrices,
reducing the number of global memory accesses and improving performance.
2. **Stencil Computations**: In algorithms that require neighbouring data
points, shared memory can store these points, allowing for faster access.
3. **Reduction Operations**: Shared memory is often used in parallel reduction
algorithms to combine results from multiple threads efficiently.

Benefits of using shared memory:
1. **Reduced Latency**: Accessing shared memory is much quicker than accessing
global memory, reducing the latency in data retrieval.
2. **Efficient Data Sharing**: It allows for efficient sharing of data among
threads within the same block, which can significantly improve performance for
certain algorithms.
3. **Synchronisation**: Threads within a block can synchronize their operations
using shared memory, ensuring that all threads have a consistent view of the
data.

Key characteristics:
1. **Speed**: Shared memory is significantly faster than global memory, which is
located off-chip. This speed advantage is due to its proximity to the GPU cores.
2. **Limited Size**: Shared memory is limited in size, typically ranging from
48KB to 96KB per block, depending on the GPU architecture.
3. **Scope**: It is only accessible by threads within the same block, making it
ideal for data that needs to be shared among these threads.

Note: in the above example we have fixed the size of the `tmp`
object at compile time ("static" shared memory).


---
# Synchronisation

There are quite a large number of synchronisation options for
threads within a block in HIP. The essential one is probably
```cpp
  __syncthreads();
```
This is a barrier-like synchronisation which says that all
the threads in the block must reach the `__syncthreads()`
statement before any are allowed to continue.


---
# Example
Here is a (slightly contrived) example:
```cpp
/* Reverse elements so that the order 0,1,2,3,...
 * becomes ...,3,2,1,0
 * Assume we have one block. */

__global__ void reverseElements(int *myArray) {

  __shared__ int tmp[THREADS_PER_BLOCK];

  int idx = threadIdx.x;
  tmp[idx] = myArray[idx];

  __syncthreads();

  myArray[THREADS_PER_BLOCK - (idx+1)] = tmp[idx];
}
```

---
# Synchronisation hazards

The usual considerations apply when thinking about thread
synchronisation. E.g.,
```cpp
   if (condition) {
      __syncthreads();
   }
```
There is a potential for deadlock.


---
# Branch divergence

Branch divergence can significantly impact GPU performance due to the way GPUs
execute instructions. Branch divergence occurs when different threads within the same warp (a group of threads that execute instructions in lock-step) take different execution paths due to conditional statements like `if-else`.

It is beneficial for performance to avoid "wavefront divergence"
e.g.,
```cpp
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid % 2 == 0) {
    /* threads 0, 2, 4, ... */
  }
  else {
    /* threads 1, 3, 5 ... *.
  }
```
may cause serialisation. If half the threads in a warp process even numbers and the other half process odd numbers, the warp will first execute the instructions for even numbers while the odd-number threads are idle, and then execute the instructions for odd numbers while the even-number threads are idle. This results in inefficient use of the GPU's resources.

For this reason you may see things like
```c
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if ((tid / warpSize) % 2 == 0) {
     /* threads 0, 1, 2, ... */
  }
  else {
     /* threads 32, 33, 34, ... */
  }
```
where `warpSize`(a.k.a. wavesize) is another special value provided by HIP/CUDA.

To mitigate branch divergence, organize data such that threads within the same warp are more likely to follow the same execution path. Where possible, reduce the use of conditional statements within kernels.


---
# Other potential performance concerns

Shared memory via `__shared__` is a finite resource. The exact amount
will depend on the particular hardware, but may be in the region of
64 KB. (A portable program might have to take action at run time to
control this: e.g., using "dynamic" shared memory, where the size is
set as a kernel launch parameter.)

If an individual block requires a large amount of shared memory, then
this may limit the number of blocks that can be scheduled at the same
time, and so harm occupancy.


---
# Exercise (20 minutes)

In the following exercise we we implement a vector scalar product
in the style of the BLAS level 1 routine `ddot()`.

The template provided sets up two vectors `x` and `y` with some
initial values. The exercise is to complete the `ddot()` kernel
which we will give the prototype:
```c
  __global__ void ddot(int n, double *x, double *y, double *result);
```
where the `result` is a single scalar value which is the dot
product. A naive serial kernel is provided to give the correct
result.

Suggested procedure
1. Use a `__shared__` temporary variable to store the contribution from
each different thread in a block, and then compute the sum for the block.
2. Accumulate the sum from each block to the final answer.

Remember to deal correctly with any array 'tail'.

Some care may be needed to check the results for this problem. For
debugging, one may want to reduce the problem size; however, there
is the chance that an erroneous code actually gives the expected
answer by chance. Be sure to check with a larger problem size.


---
# Finished?

It is possible to use solely `atomicAdd()` to form the result (and not
do anything using `__shared__` within a block)? Investigate the performance
implications of this (particularly, if the problem size becomes larger).
You will need two versions of the kernel.
