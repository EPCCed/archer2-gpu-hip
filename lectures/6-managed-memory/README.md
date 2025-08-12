template: titleslide
# Managed Memory



---
# Explicit memory allocation

We have seen an explicit approach using standard C pointers.

```cpp
double *h_ptr = NULL;
double *d_ptr = NULL;

h_ptr = (double *) malloc(nbytes);

hipMalloc(&d_ptr, nbytes);
hipMemcpy(d_ptr, h_ptr, nbytes, hipMemcpyHostToDevice);
```

The host pointer to the device memory is then used in the kernel invocation.

```cpp
myKernel<<<...>>>(d_ptr);
```

Such code is perfectly fine, particularly if it is only used to transfer large contiguous blocks of data.



---
# Explicit memory allocation

We have seen an explicit approach using standard C pointers.

```cpp
double *h_ptr = NULL;
double *d_ptr = NULL;

h_ptr = (double *) malloc(nbytes);

hipMalloc(&d_ptr, nbytes);
hipMemcpy(d_ptr, h_ptr, nbytes, hipMemcpyHostToDevice);
```

The host pointer to the device memory is then used in the kernel invocation.

```cpp
myKernel<<<...>>>(d_ptr);
```

Such code is perfectly fine, particularly if it is only used to transfer large contiguous blocks of data.

However, pointers to device memory cannot be dereferenced on the host, and there is a need to have both a host reference (`h_ptr`)
and a device reference (`d_ptr`) in the code.

This can become onerous when developing code that uses complex data access patterns.



---
# Managed memory

Managed memory establishes an effective single reference to memory which can be accessed on both host and device.

```cpp
__host__ hipError_t hipMallocManaged(void **ptr, size_t sz);
```

Host/device transfers are managed automatically as the need arises.

```cpp
double *ptr = NULL;

hipMallocManaged(&ptr, nbytes);

/* Initialise values on host ... */

for (int i = 0; i < ndata; i++) {
  ptr[i] = 1.0;
}

/* Use data in a kernel ... */
kernel<<<...>>>(ptr);
```

The single call to `hipMallocManaged()` replaces the usual combination of `malloc()` <br>(or `new` in C++) and `hipMalloc()`.



---
# Managed memory

It is often useful to start development with managed memory and then move to explicit `malloc()/hipMalloc()` calls if it is required for performance.

<br>
Managed memory is released in the same way as memory allocated via `hipMalloc()`.

```cpp
hipFree(ptr);
```



---
# Managed memory requires page migration

- Managed data is transferred between host and device via page migration.
  - A page is the smallest unit of memory management and is often 4096 bytes on a typical (CPU) machine.



---
# Managed memory requires page migration

- <span style="color:gray">Managed data is transferred between host and device via page migration.</span>
  - <span style="color:gray">A page is the smallest unit of memory management and is often 4096 bytes on a typical (CPU) machine.</span>

- Assume that some code `hipMallocManaged()` allocates memory in the host space, initialises memory on the host and calls a kernel.
  - When the GPU starts executing the kernel, any access to the relevant (virtual) address is not present on the GPU, prompting a page fault.
  - The relevant page of memory must be migrated (i.e. copied) from the host to the GPU before useful execution can continue.



---
# Managed memory requires page migration

- <span style="color:gray">Managed data is transferred between host and device via page migration.</span>
  - <span style="color:gray">A page is the smallest unit of memory management and is often 4096 bytes on a typical (CPU) machine.</span>

- <span style="color:gray">Assume that some code `hipMallocManaged()` allocates memory in the host space, initialises memory on the host and calls a kernel.</span>
  - <span style="color:gray">When the GPU starts executing the kernel, any access to the relevant (virtual) address is not present on the GPU, prompting a page fault.</span>
  - <span style="color:gray">The relevant page of memory must be migrated (i.e. copied) from the host to the GPU before useful execution can continue.</span>

- Similarly, if the same data is required by the host after the kernel finishes execution, the subsequent access request from the host
will trigger a page fault on the CPU.
  - The relevant page is copied back from the GPU to the host.



---
# Prefetching

If the programmer knows in advance that memory is required on the device before kernel execution, a prefetch to the destination device may be issued.

```cpp
hipGetDevice(&device);
hipMallocManaged(&ptr, nbytes);

/* ... initialise data ... */

hipMemPrefetchAsync(ptr, nbytes, device);

/* ... kernel activity ... */
```

As the name suggests, this is an asynchronous call (it is likely to return before any data transfer has actually occurred).
The call can be viewed as a request to the HIP run-time to transfer the data.

Prefetches from the device to the host can be requested by using the special destination value `hipCpuDeviceId`.

**Note:** Currently, the `hipMemPrefetchAsync` API is implemented on Linux and is under development for Windows.



---
# Providing hints

It is also possible to "advise" the HIP runtime re memory access patterns.

```cpp
__host__ hipError_t hipMemAdvise(const void *ptr, size_t sz,
                                 hipMemoryAdvise advice, int device);
```

**`hipMemAdviseSetReadMostly`**<br>
Data will mostly be read and only occasionally written.

**`hipMemAdviseSetPreferredLocation`**<br>
Set the preferred location for the data as the specified device.

**`hipMemAdviseSetAccessedBy`**<br>
Data will be accessed by the specified device so prevent page faults as much as possible.

**`hipMemAdviseSetCoarseGrain`**<br>
The default fine-grain memory model allows coherent operations between host and device, while executing kernels. The coarse-grain model can be used for data that only needs to be coherent at dispatch boundaries for better performance.



---
# Providing hints (continued)

Each `hipMemoryAdvise` option has an `Unset` twin (e.g. `hipMemAdviseSetReadMostly` and `hipMemAdviseUnsetReadMostly`) that nullifies the effect of a preceding `Set`.

Further info can be found in the [HIP Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___global_defs.html).


**Note:** Currently, the `hipMemAdvise` API and `hipMemoryAdvise` are implemented on Linux and are under development for Windows.



---
# <span style="color:red">Exercise:</span> Managed memory

A solution to the previous exercise that used explicit host/device memory management
is provided by the [exercise_dger.hip.cpp](../../exercises/6-managed-memory/1-managed-memory/) template.

That exercise computed the matrix operation, A<sub>ij</sub> = A<sub>ij</sub> + α x<sub>i</sub> y<sub>j</sub>.


Confirm you can replace the explicit memory management using `new/hipMalloc()` and `hipMemcpy()` with managed memory.
It is suggested that both "`d_a`" and "`h_a`" are replaced by the single declaration "`a`" in the main function.

Run the new code to check the answers are correct and make a note of the new output of `rocprof` associated with managed memory.

Add the relevant prefetch requests for the vectors "`x`" and "`y`" before the kernel, and the matrix "`a`" after the kernel.
Note that the device id is already present in the code as `deviceNum`.



---
# <span style="color:green">Finished?</span>

What happens if you should accidentally use `hipMalloc()` where you intended to use `hipMallocManaged()`?



---
# <span style="color:red">Next Lecture</span>

<br>
## [Shared Memory](../7-shared-memory)