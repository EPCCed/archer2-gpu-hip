template: titleslide
# HIP Programming



---
# HIP programming

We will assume that there are separate address spaces for CPU and GPU memory.

.center[![:scale_img 70%](separate_memory_spaces.svg)]

This means we will need to consider how to move data between the two memory spaces.



---
# HIP programming

We will assume that there are separate address spaces for CPU and GPU memory.

.center[![:scale_img 70%](separate_memory_spaces.svg)]

This means we will need to consider how to move data between the two memory spaces.

![:thumb](The latest GPUs such as the AMD MI300A and NVIDIA GH200 feature <br>*unified* memory, i.e. a single address space shared by the host and device.)



---
# What to include and what not to include

A standard C/C++ source may start as below.

```cpp
#include "hip/hip_runtime.h"
```

This is usually relevant for programs to be compiled by `hipcc`.

There is also a subset runtime.

```cpp
#include "hip_runtime_api.h"
```

This is the C/C++ interface which does not need to be compiled with `hipcc`.



---
# Memory management

Data accessed by kernels must reside in device memory, sometimes also
referred to as *device global memory*, or just *global memory*.

There are different ways of managing the allocation and movement
of data between host and device.

1. Explicit allocations and explicit copies.
2. Use of *managed* memory.

We will look at the explicit way first.


---
# Memory allocation

A variable that references device memory is declared using the standard C/C++ data types and pointers.

```cpp
double *data = NULL;   /* Device data */

err = hipMalloc(&data, nArray*sizeof(double));

/* ... perform some work ... */

err = hipFree(data);
```

Such pointers are "host pointers to device memory". They have a value,
but cannot be dereferenced in host code, which will cause a runtime error.



---
# Memory movement

Assuming we have established some data on the host, the copying of data to and from the device is done via `hipMemcpy()`.

```cpp
err = hipMemcpy(data, hostdata, nArray*sizeof(double),
                hipMemcpyHostToDevice);

/* ... do something ... */

err = hipMemcpy(hostdata, data, nArray*sizeof(double),
                hipMemcpyDeviceToHost);
```

These are *blocking* calls: they will not return until the transfer of data has been
completed (or an error has occurred).

<br>
The `hipMemcpy()` prototype is as follows.

```cpp
hipError_t hipMemcpy(void *dest, void *src, size_t sz,
                     hipMemcpyKind direction);
```



---
# Error handling

Most HIP API routines return an error code of type `hipError_t`. <br>
It is important to check this code against `hipSuccess`.

If an error occurs, the returned code can be interrogated to provide
some meaningful information.

```cpp
const char *hipGetErrorName(hipError_t err);    /* Name */
const char *hipGetErrorString(hipError_t err);  /* Descriptive string */
```



---
# Error handling in practice

The requirement for error handling often appears in real code as a macro.

```cpp
HIP_ASSERT( hipMalloc( &data, nArray*sizeof(double) ) );
```

To avoid clutter, we omit this error checking in the example code snippets.

However, for the code exercises, we have provided such a macro, and
it should be used.

It is particularly important to check the result of the first API call in the code.
This will detect any problems with the HIP context, which might otherwise manifest
later on in the code as apparently unrelated runtime errors.


---
# <span style="color:red">Exercise:</span> Data transfer (1/7)

Read the [`exercise_dscal.hip.cpp`](../../exercises/3-hip-programming/1-data-transfer/) souce file. It is a template for implementing a simple scale function that multiplies all the
elements of an array by a constant.

First, check you can compile and run the unaltered template code.

To use the AMD compilation suite, please load the modules below.

```bash
module load PrgEnv-amd
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan
```

The compilation command is as follows.

```bash
hipcc -x hip -std=c++11 --rocm-path=${ROCM_PATH} \
      -D__HIP_ROCclr__ -D__HIP_PLATFORM_AMD__ \
      --offload-arch=gfx90a exercise_dscal.hip.cpp
```

Once the code has compiled, please submit to the queue system using the [`submit.sh`](../../exercises/3-hip-programming/1-data-transfer/) script.



---
# <span style="color:red">Exercise:</span> Data transfer (2/7)

If the code has run correctly, you should see something like the output below.

```
Device 0 name: 
Maximum number of threads per block: 1024
Results:
No. elements 256, and correct: 1
```

Only one output array element is correct.



---
# <span style="color:red">Exercise:</span> Data transfer (3/7)

Now add the code that allocates and moves data to and from the GPU.

1. Declare and allocate device memory (call it "`d_x`") of type `double`.
2. Copy the initialised host array "`h_x`" to device array "`d_x`".
3. Copy the (unaltered) device array "`d_x`" back to the host array "`h_out`" and check that "`h_out`" has the expected values.
4. Release the device memory "`d_x`" at the end of execution.

Remember, use the `HIP_ASSERT` macro to check the return values of the HIP API calls.

As there is no kernel yet (this will added in the next exercise), the output "`h_out`" should
just be the same as the input "`h_in`".

All 256 array elements should be correct.



---
# <span style="color:green">Finished?</span>

Check the HIP documentation to see what other information is available from the structure type [`hipDeviceProp_t`](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/structhip_device_prop__t.html#).

What other possibilities exist for [`hipMemcpyKind`](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___driver_types.html#ga232e222db36b1fc672ba98054d036a18)?

<br>
Click [https://rocm.docs.amd.com/projects/HIP/en/latest/index.html](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html) for HIP documentation.

<br>
What happens if you confuse the order of the host and device references in a call to `hipMemcpy()`?

```cpp
hipMemcpy(hostdata, devicedata, sz, hipMemcpyHostToDevice);
```



---
# <span style="color:red">Next Lecture</span>

<br>
## [Kernels](../4-kernels)