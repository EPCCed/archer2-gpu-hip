template: titleslide
# Constant Memory



---
# Constant memory

One further type of memory used in HIP programs is *constant* memory, which is device memory for storing values
that cannot be updated for the duration of a kernel.

Physically, this is likely to be a small cache on each GPU Compute Unit (CU).
It provides fast *read-only* access to frequently used values.

Constant memory is a limited resource &mdash; the exact size varies with hardware.



---
# Constant memory characteristics

1. **Read-Only**: Constant memory is read-only for the GPU kernels. It is written to by the host (CPU) before the kernel execution.

2. **Cached**: It is cached on-chip, which means that access to constant memory is much faster than access to global memory.

3. **Limited size**: The size of constant memory is limited, typically around 64 KB, depending on the GPU architecture.



---
# Constant memory benefits

1. **Broadcasting**: When all threads in a wavefront access the same address in constant memory, the value is broadcast to all threads, resulting in a single memory read.

2. **Reduced latency**: Accessing constant memory is faster than accessing global memory due to constant memory being on-chip.

3. **Efficiency**: It is ideal for storing constants that are used by all threads, such as coefficients in mathematical formulae.



---
# Constant memory use cases

1. **Lookup tables**: Storing lookup tables that are frequently accessed by the kernel.

2. **Constants**: Storing constants that are used across many threads, such as physical constants or coefficients.

3. **Configuration data**: Storing configuration parameters that do not change during kernel execution.



---
# Kernel parameters

If one calls a kernel function, actual arguments are (conceptually, at least) passed by value as in standard C++, and are placed in constant memory.

```c
__global__ void kernel(double arg1, double *arg2, ...);
```

Note, constant memory may be small in size (e.g. 4096 bytes), so large objects should not be passed by value to the device.



---
# Constant memory at file scope

It is also possible to use the `__constant__` memory space qualifier for objects declared at file scope.

```c
static __constant__ double data_read_only[3];
```

Host values can be copied to the device with the `hipMemcpyToSymbol()` function.

```c
double values[3] = {1.0, 2.0, 3.0};

hipMemcpyToSymbol(data_read_only, values, 3*sizeof(double));
```

The object `data_read_only` may then be accessed by a kernel or kernels at the same scope.

Again, capacity is limited; so, if an object is too large it will probably spill into global memory.



---
# <span style="color:red">Exercise:</span> Matrix-vector product (1/2)

We should now be in a position to combine our matrix operation and the reduction required for the vector product to perform another
useful operation, a matrix-vector product.

For a matrix A<sub>mn</sub> of `m` rows and `n` columns, the product,
<br><br>
y<sub>i</sub> = α A<sub>ij</sub> x<sub>j</sub> ,
<br><br>
may be formed with a vector `x` of length `n` to give a result `y` of length `m`.<br>α is a constant.

A new template has been provided ([exercise_dgemv.hip.cpp](../../exercises/8-constant-memory/1-matrix-vector)), one that implements a simple serial version.


---
# <span style="color:red">Exercise:</span> Matrix-vector product (1/2)

1. To start, make the simplifying assumption that we have only 1 block per row, and that the number of columns is equal to
the number of threads per block. This should allow the elimination of the loop over both rows and columns with judicious choice of thread indices.

2. The limitation to one block per row may harm occupancy. So, we need to generalise to allow columns to be distributed between
different blocks.<br><br>**Hint:** you will probably need a two-dimensional `__shared__` provision in the kernel. Use the same total number
of threads per block with `blockDim.x == blockDim.y`. Make sure you can increase the problem size (specifically, the number
of columns `ncol`) and retain the correct answer.

3. Leave the concern of coalescing until last. The indexing can be rather confusing. Again, remember to deal with any array "tails".



---
# <span style="color:green">Finished?</span>

A fully robust solution might check the result with a rectangular thread block.

A level 2 BLAS implementation may want to compute the update, y<sub>i</sub> = α A<sub>ij</sub> x<sub>j</sub> + β y<sub>i</sub>.

How does this complicate the simple matrix-vector update?



---
# <span style="color:red">Next Lecture</span>

<br>
## [Profiling](../9-profiling)