template: titleslide
# Shared Memory


---
# Shared memory

GPUs have global memory and shared memory
The former is allocated on the host and accessed from the kernel.

Variables local to the kernel, appear on the stack in the expected fashion.
These are stored in registers and have distinct values for each thread.

```cpp
int i = blockDim.x*blockIdx.x + threadIdx.x;
```

While global memory is shared between all threads, the usage of "shared memory" is reserved for something
more specific in the GPU context.



---
# Independent accesses to global memory

So far, kernels have been used to replace loops that have no dependencies between iterations.

```cpp
for (int i = 0; i < ndata; i++) {
  data[i] = 2.0*data[i];
}
```

For example, the loop above can be replaced by a kernel containing the loop body.

```cpp
__global__ myKernel(int ndata, double *data) {
  int i = blockDim.x*blockIdx.x + threadIdx.x;

  data[i] = 2.0*data[i];
}
```

As each thread accesses an independent location in (global) memory, there are no potential conflicts.



---
# Accessing the same global memory address

Consider a loop that accumulates a value.

```cpp
double sum = 0.0;

for (int i = 0; i < ndata; i++) {
  sum += data[i];
}
```

The iterations are now coupled in some sense: all threads must add a value to the single memory location `sum`.

What would happen if we tried to run a kernel of the following form?

```cpp
__global__ myKernel(int ndata, double *data, double *sum) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;

  if (i < ndata) *sum += data[i];
}
```



---
# Race condition

The value of `sum` will differ from run to run. Such non-deterministic behaviour is referred to as a race condition.

The problem lies in the fact that the increment is actually a number of different operations which need to occur in a specific order.

```cpp
*sum += ndata[i];
```

1. Read the current value of `sum` from memory into a register.
2. Undertake the appropriate arithmetic in register.
3. Store the new result back to global memory.

Unexpected results can arise if many threads are performing these operations in an uncontrolled fashion.



---
# Synchronisation

In practice, potentially unsafe updates to any form of shared memory can be protected by appropriate synchronisation,
i.e. guarantees that operations happen in the correct order.

For global memory, we require a so-called *atomic* update.

```cpp
/* *sum += data[i]; */  
atomicAdd(sum, data[i]); 
```

Such updates are usually implemented by some form of lock.

The atomic update is a single unified operation on a single thread.

1. Obtain a lock on the relevant memory location (`sum`).
2. Read the existing value into register and update.
3. Store the result back to the global memory location.
4. Release the lock on that location.

Obtaining the lock blocks other threads from updating `sum`.



---
# Atomic functions

The are many variants of atomic functions, covering a range of operations
and data types.

For example, `atomicAdd()` is an overloaded device function.

```cpp
__device__ int atomicAdd(int *address, int value);
__device__ double atomicAdd(double *address, double value);
...
```

The value returned is the old value of the target variable.

Click the link below for further reading.<br><br>
[https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#atomic-functions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#atomic-functions)



---
# (Block) Shared memory

Shared memory can be declared explicitly on the GPU using the `__shared__` memory space qualifier.

```cpp
__shared__ double tmp[THREADS_PER_BLOCK];
```

This shared memory is accessible by all threads within the same block, i.e. those threads that
have been allocated to the same GPU Compute Unit (CU).

It is faster to access than global memory because it is located on-chip, close to the CU cores.
This makes it ideal for storing data that needs to be frequently accessed by multiple threads.



---
# (Block) Shared memory

Shared memory can be declared explicitly on the GPU using the `__shared__` memory space qualifier.

```cpp
__shared__ double tmp[THREADS_PER_BLOCK];
```

This shared memory is accessible by all threads within the same block, i.e. those threads that
have been allocated to the same GPU Compute Unit (CU).

It is faster to access than global memory because it is located on-chip, close to the CU cores.
This makes it ideal for storing data that needs to be frequently accessed by multiple threads.


![:thumb](Note, in the code snippet above, we have fixed the size of the `tmp` object at compile time, making
it an example of *static* shared memory.)



---
# Shared memory characteristics

1. **Speed**: Shared memory is significantly faster than global memory, which is
located off-chip. This speed advantage is due to its proximity to the GPU cores.

2. **Limited Size**: Shared memory is limited in size, typically ranging from
48KB to 96KB per block, depending on the GPU architecture.

3. **Scope**: It is only accessible by threads within the same block, making it
ideal for data that needs to be shared among these threads.



---
# Shared memory benefits

1. **Reduced Latency**: Accessing shared memory is much quicker than accessing
global memory, reducing the latency in data retrieval.

2. **Efficient Data Sharing**: It allows for efficient sharing of data among
threads within the same block, which can significantly improve performance for
certain algorithms.

3. **Synchronisation**: Threads within a block can synchronize their operations
using shared memory, ensuring that all threads have a consistent view of the
data.



---
# Shared memory use cases

1. **Matrix Multiplication**: Shared memory can be used to store sub-matrices,
reducing the number of global memory accesses and improving performance.

2. **Stencil Computations**: In algorithms that require neighbouring data
points, shared memory can store these points, allowing for faster access.

3. **Reduction Operations**: Shared memory is often used in parallel reduction
algorithms to combine results from multiple threads efficiently.



---
# Explicit Synchronisation

In HIP, there are many synchronisation options for threads within a block. The main one however
is `__syncthreads()`.

This is a barrier-like synchronisation which says that all the threads in the block must reach
the `__syncthreads()` call before any are allowed to continue.



---
# Synchronisation example

```cpp
/* Reverse elements so that the order 0,1,2,3,... */
/* becomes ...,3,2,1,0 */
/* Assume we have one block. */

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

The usual considerations apply when thinking about thread synchronisation.

```cpp
if (condition) {
  __syncthreads();
}
```

Deadlock will occur if one or more threads cannot reach the synchronisation call.



---
# Branch divergence

Branch divergence occurs when different threads within the same wavefront (a group of threads that execute the same instructions in lock-step) take different execution paths.

```cpp
  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid % 2 == 0) {
    /* threads 0, 2, 4, ... */
  }
  else {
    /* threads 1, 3, 5 ... */
  }
```

In the example above, the even numbered threads do work while the odd numbered threads are kept idle, and vice versa.<br>

All this causes GPU resources to be used inefficiently.



---
# Branch divergence mitigation

We can mitigate branch divergence by altering the code such that threads within the same warp are more likely to follow the same execution path.

```c
int tid = blockIdx.x*blockDim.x + threadIdx.x;

if ((tid / warpSize) % 2 == 0) {
   /* threads 0, 1, 2, ... */
}
else {
   /* threads 32, 33, 34, ... */
}
```

Here, `warpSize` is the number of threads within the wavefront &mdash; it is provided by HIP/CUDA.



---
# Branch divergence mitigation

We can mitigate branch divergence by altering the code such that threads within the same warp are more likely to follow the same execution path.

```c
int tid = blockIdx.x*blockDim.x + threadIdx.x;

if ((tid / warpSize) % 2 == 0) {
   /* threads 0, 1, 2, ... */
}
else {
   /* threads 32, 33, 34, ... */
}
```

Here, `warpSize` is the number of threads within the wavefront &mdash; it is provided by HIP/CUDA.

![:thumb](In general, reduce the use of conditional statements within kernels where possible.)



---
# Other potential performance concerns

- Shared memory via `__shared__` is a finite resource. The exact amount will depend on the particular hardware &mdash; it is usually in the region of 64 KB. 

- A portable program might have to take action at run time to control this, e.g. using "dynamic" shared memory, where the size is set as a kernel launch parameter.

- If an individual block requires a large amount of shared memory, this may limit the number of blocks that can be scheduled at the same time, which in turn limits occupancy.



---
# <span style="color:red">Exercise:</span> Vector scalar (1/2)

In this exercise we implement a vector scalar product in the style of the BLAS level 1 <br>routine, `ddot()`.

The template provided sets up two vectors, "`x`" and "`y`" with some initial values. The aim is to complete the `ddot()` kernel
following the prototype below.

```c
__global__ void ddot(int n, double *x, double *y, double *result);
```

The `result` parameter is a single scalar value which stores the dot product.

A naive serial kernel is provided to give the correct result, see the code template, [exercise_ddot.hip.cpp](../../exercises/7-shared-memory/1-vector-scalar).



---
# <span style="color:red">Exercise:</span> Vector scalar (2/2)

The suggested procedure has two steps.

1. Use a `__shared__` temporary variable to store the contribution from
each different thread in a block and then compute the sum for the block.

2. Accumulate the sum from each block to the final answer.

Remember to deal correctly with any array "tail".

Some care may be needed to check the results. For debugging, one may want to reduce the problem size;
however, there is the chance that an erroneous code actually gives the expected answer by chance, so be sure to check
with a larger problem size.



---
# <span style="color:green">Finished?</span>

It is possible to use solely `atomicAdd()` to form the result (and not do anything using `__shared__` within a block)?

Investigate the performance implications of this (particularly, if the problem size becomes larger).
You will need two versions of the kernel.



---
# <span style="color:red">Next Lecture</span>

<br>
## [Constant Memory](../8-constant-memory)