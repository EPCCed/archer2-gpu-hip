template: titleslide
# Kernels



---
# Kernels

- In many scientific applications, a large fraction of the computational effort exists within discrete sections of code, otherwise known as *kernels*.

- It is natural then that kernels should be considered first for offloading to GPU.

- Kernels are often associated with loops which have independent iterations that can be distributed across threads running in parallel.



---
# A simple example

The iterations of the loop below are independent of each other.

```cpp
for (int i = 0; i < ARRAY_LENGTH; i++) {
  result[i] = 2*i;
}
```

We can therefore safely run the iterations in parallel.



---
# A simple example

The iterations of the loop below are independent of each other.

```cpp
for (int i = 0; i < ARRAY_LENGTH; i++) {
  result[i] = 2*i;
}
```

We can therefore safely run the iterations in parallel.

In HIP, this can be done by first defining a kernel that replicates the loop body.

```cpp
__global__ void myKernel(int *result) {

  int i = threadIdx.x;
  result[i] = 2*i;

}
```

The entry point for GPU execution is denoted by the `__global__` execution space qualifier.

HIP allows us to identify the thread by providing a special variable, `threadIdx.x`.



---
# Invoking the kernel

We next add the code that *launches* the kernel on the GPU.

```cpp
dim3 blocks = {1, 1, 1};
dim3 threadsPerBlock = {ARRAY_LENGTH, 1, 1};

myKernel<<<blocks, threadsPerBlock>>>(result);
```

The language extension `<<<...>>>`, placed between the kernel's name and arguments,
encapsulates the execution configuration, within which we specify the *number of blocks* <br>
and the *number of threads per block* for three dimensions.



---
# Invoking the kernel

We next add the code that *launches* the kernel on the GPU.

```cpp
dim3 blocks = {1, 1, 1};
dim3 threadsPerBlock = {ARRAY_LENGTH, 1, 1};

myKernel<<<blocks, threadsPerBlock>>>(result);
```

The language extension `<<<...>>>`, placed between the kernel's name and arguments,
encapsulates the execution configuration, within which we specify the *number of blocks* <br>
and the *number of threads per block* for three dimensions.

In this example, the problem size is `ARRAY_LENGTH`, which must match the total number <br>
of threads specified, `1 x ARRAY_LENGTH`.


---
# Invoking the kernel

We next add the code that *launches* the kernel on the GPU.

```cpp
dim3 blocks = {1, 1, 1};
dim3 threadsPerBlock = {ARRAY_LENGTH, 1, 1};

myKernel<<<blocks, threadsPerBlock>>>(result);
```

The language extension `<<<...>>>`, placed between the kernel's name and arguments,
encapsulates the execution configuration, within which we specify the *number of blocks* <br>
and the *number of threads per block* for three dimensions.

The `dim3` type provided by HIP is defined like so.

```cpp
struct {
  unsigned int x;
  unsigned int y;
  unsigned int z;
} dim3;
```

Variables of type `dim3` can be initialised in C as above, or using C++ style constructors.


---
# More than one block

A larger problem requires a larger array and so more blocks are needed.

```cpp
__global__ void myKernel(int *result) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  result[i] = 2*i;
  
}
```

For calling the kernel, we assume `ARRAY_LENGTH` is a multiple of `THREADS_PER_BLOCK`.

```cpp
threadsPerBlock.x = THREADS_PER_BLOCK;
blocks.x          = ARRAY_LENGTH/THREADS_PER_BLOCK;

myKernel<<< blocks, threadsPerBlock >>>(result);
```



---
# More than one block

A larger problem requires a larger array and so more blocks are needed.

```cpp
__global__ void myKernel(int *result) {

  int i = blockIdx.x*blockDim.x + threadIdx.x;
  result[i] = 2*i;

}
```

For calling the kernel, we assume `ARRAY_LENGTH` is a multiple of `THREADS_PER_BLOCK`.

```cpp
threadsPerBlock.x = THREADS_PER_BLOCK;
blocks.x          = ARRAY_LENGTH/THREADS_PER_BLOCK;

myKernel<<< blocks, threadsPerBlock >>>(result);
```

<br>
In general,
- choose a number of threads per block that is a power of two (e.g. 128, 256, ..., 1024);
- then set the number of blocks sufficient for the problem space.



---
# Internal variables in the kernel

A number of internal variables are made available by the HIP runtime and can be used
in the kernel to locate a given thread's position within in the grid.

```cpp
dim3 gridDim;     /* The number of blocks */
dim3 blockDim;    /* The number of threads per block */

/* Unique to each block */
dim3 blockIdx;    /* 0 <= blockIdx.x < gridDim.x  etc. for y,z */

/* Unique to each thread (within a block) */
dim3 threadIdx;   /* 0 <= threadIdx.x < blockDim.x  etc. for y,z */
```

These variable names should be considered reserved.



---
# Synchronisation between host and device

Kernel launches are asynchronous on the host, i.e. the launch call returns immediately. 

```cpp
myKernel<<< blocks, threadsPerBlock >>>(arg1, arg2, arg3);

/* ... returns immediately */

hipError_t err = hipDeviceSynchronize();

/* ... it is now safe to use the results of the kernel ... */
```

In order to be sure that the kernel has actually completed, we need synchronisation.



---
# Error handling

Errors occurring in the kernel execution are also asynchronous, which can cause some confusion.
As a result, one will sometimes see this usage.

```cpp
myKernel<<< blocks, threadsPerBlock >>>(arg1, arg2, arg3);

HIP_ASSERT( hipPeekAtLastError() );
HIP_ASSERT( hipDeviceSynchronize() );
```



---
# Error handling

Errors occurring in the kernel execution are also asynchronous, which can cause some confusion.
As a result, one will sometimes see this usage.

```cpp
myKernel<<< blocks, threadsPerBlock >>>(arg1, arg2, arg3);

HIP_ASSERT( hipPeekAtLastError() );
HIP_ASSERT( hipDeviceSynchronize() );
```
<br>
The first function, `hipPeekAtLastError()`, queries the error state without altering it
and will fail if there are errors in the kernel launch, 
  - e.g. the GPU is not available or there are errors in the configuration.



---
# Error handling

Errors occurring in the kernel execution are also asynchronous, which can cause some confusion.
As a result, one will sometimes see this usage.

```cpp
myKernel<<< blocks, threadsPerBlock >>>(arg1, arg2, arg3);

HIP_ASSERT( hipPeekAtLastError() );
HIP_ASSERT( hipDeviceSynchronize() );
```
<br>
The first function, `hipPeekAtLastError()`, queries the error state without altering it
and will fail if there are errors in the kernel launch, 
  - e.g. the GPU is not available or there are errors in the configuration.

<br>
Errors that occur during the execution of the kernel itself will not be apparent until `hipDeviceSynchronize()`.
  - These are typically due to programmer error and will require debugging.


---
# An alternative method for launching kernels

If you are not keen on the non-standard looking execution configuration `<<<...>>>`,
one can instead use the `hipLaunchKernelGGL()` macro. 

```cpp
hipError_t hipLaunchKernelGGL(const void *func, dim3 blocks, dim3 threads, 
                              size_t dynamicShared, hipStream_t stream,
                              args...);
```

The kernel function is `func`, while the second and third arguments are the *number of blocks*
and *threads per block*. The fourth argument holds the amount of additional shared memory to
allocate when launching the kernel, and the fifth argument is the stream where the kernel should
execute. The arguments following (`args`) are those passed to the kernel itself.

The `myKernel` kernel from earlier is therefore called like so.

```cpp
hipLaunchKernelGGL(myKernel, blocks, threadsPerBlock, 0, 0, result);
```



---
# <span style="color:red">Exercise:</span> Apply a scalar (1/4)

Starting from your solution to the previous exercise, we will now add the relevant kernel and execution configuration. You should adjust the value of the constant, e.g. `a = 2.0`.

There is also a new [`exercise_dscal.hip.cpp`](../../exercises/4-kernels/1-data-transfer/) template with a canned solution to the previous exercise.


---
# <span style="color:red">Exercise:</span> Apply a scalar (2/4)

Write a kernel of the prototype scale function.

```cpp
__global__ void myKernel(double a, double *x);
```

It performs the scale operation on a given element of the array.
Limit yourself to one block in the first instance (you only need `threadIdx.x`).

Next, in the main part of the program, declare and initialise variables of type `dim3`
to hold the *number of blocks* and the *number of threads per block*.

You can use one block and `THREADS_PER_BLOCK` in the first instance.


---
# <span style="color:red">Exercise:</span> Apply a scalar (3/4)

Update the kernel and then the execution configuration parameters to allow more than one block.
We will keep the assumption that the array length is a whole number of blocks.

Increase the array size `ARRAY_LENGTH` to 512 and check you retain the correct behaviour.
<br><br>
Check for larger multiples of `THREADS_PER_BLOCK`.


---
# <span style="color:red">Exercise:</span> Apply a scalar (4/4)

As we are effectively constrained in the choice of `THREADS_PER_BLOCK`, it is likely that
in general, the problem space will not be an integral number of blocks.

How can we deal with this situation?

1. For the launch parameters, you will need to compute a number of blocks that is sufficient and necessary to cover the entire problem space. There needs to be at least one block, but no more than necessary.

2. You will also need to make an adjustment in the kernel. To avoid what type of error?

Set the array size to 100 and then 1000, checking your results.



---
# <span style="color:green">Finished?</span>


All kernels must be declared `void`. Why do you think this is the case?


Adapt your program to try another simple level one BLAS routine, which
we will take to have the prototype:
```cpp
void daxpy(int nlen, double a, const double * x, double * y);
```
This is the operation `y = ax + y` where "`y`" is incremented, and "`x`" is
unchanged. <br>Both vectors have length "`nlen`".



---
# <span style="color:red">Next Lecture</span>

<br>
## [Performance](../5-performance)