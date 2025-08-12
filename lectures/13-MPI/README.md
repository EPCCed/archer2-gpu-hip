template: titleslide
# MPI



---
# Using MPI with GPUs

If one has more than one GPU, and certainly if one has more than one node with GPUs,
it is natural to ask how to think about programming with MPI.

First, this may require a design decision about how to go about the problem.



---
# Setting the device based on MPI rank

A natural choice may be to run one MPI process per device.

For example, on a node with 4 GPUs, we would ask for 4 MPI processes.
Each individual MPI rank would just set the current device appropriately.

```c
int rank = -1; /* MPI rank */
MPI_Comm_rank(comm, &rank);

hipSetDevice(rank % ndevicePerNode);
```

The number of devices per node may be obtained via `hipGetDeviceCount()` or it may require external input.



---
# Passing messages between devices

In order to pass a message between two devices, one could try running the code below.

```c
/* On the sending side ... */
hipMemcpy(hmsgs, dmsgs, ndata*sizeof(double), hipMemcpyDeviceToHost);
...
MPI_Isend(hmsgs, ndata, MPI_DOUBLE, dst, ...);

...

/* On the receiving side ... */
MPI_Recv(hmsgr, ndata, MPI_DOUBLE, src, ...);
...
hipMemcpy(dmsgr, hmsgr, ndata*sizeof(), hipMemcpyHostToDevice);
```

This may well lead to poor performance however.



---
# GPU-aware MPI

It is possible to use device references in MPI calls on the host.

```c
MPI_Isend(dmsgs, ndata, MPI_DOUBLE, dst, ...);
MPI_Recv(dmsgr, ndata, MPI_DOUBLE, src, ...)
```

Here, `dmsgs` and `dmsgr` are device memory references. If within a node
with fast GPU-to-GPU connections, this should be routed in the appropriate way.
A fall-back to copy via the host may be required for inter-node messages.

Some architectures have the network interface cards connected directly
to the GPUs (rather than the host), in which case, inter-node transfers would
also favour use of GPU-aware MPI.



---
# <span style="color:red">Exercise:</span> GPU-aware MPI send

The AMD ROCm software stack includes a build of OpenMPI with GPU-aware MPI enabled.

A [sample program](../../exercises/13-mpi/1-gpu-aware-mpi-send) has been provided which measures the time taken to send messages of different sizes between MPI tasks,
using the two methods outlined above.

Have a look at the program, and try to compile and run it.



---
# <span style="color:red">Final Exercise:</span> Conjugate gradient solver

<br>
## [Conjugate Gradient Solver](../14-conjugate-gradient-solver)