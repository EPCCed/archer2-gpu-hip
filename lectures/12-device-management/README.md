template: titleslide
# Device Management



---
# Available devices

The number of GPUs currently available to the host can be obtained via a simple API call.

```cpp
int ndevice = -1;
hipGetDeviceCount(&ndevice);
```

Note, the total number visible may be controlled ultimately by the scheduling system.

Devices are numbered logically, `0, 1, 2, ..., ndevice-1`.<br>
The identity of the currently "set" device, or current context, is obtained as follows.

```cpp
int myid = -1;
hipGetDevice(&myid);
```

The `hipGetDevice` function will return `0` if there is one device.
<br><br>
All following HIP API calls and kernel launches will involve this device.



---
# Multiple devices

If we have more than one GPU available to the host process, `hipGetDeviceCount()` will return the appropriate number.
The initial context however will still be device `0`, the default.

We can make use of the other devices by switching context.

```cpp
int myid1 = 1;
hipSetDevice(myid1);
```

Subsequent API calls now refer to the new device.

```cpp
double *d_data1 = NULL;
hipMalloc(&d_data1, ndata*sizeof(double));
```

For example, the above code will allocate memory on device `1`.

The same arrangement applies to kernels, i.e. a kernel is launched on the current device.



---
# Peer Access

It is perfectly valid to transfer data from one device to another, assuming
there are two memory allocations on the same node.

```cpp
hipMemcpy(d_ptr1, d_ptr2, sz, hipMemcpyDeviceToDevice);
```

More recent AMD (and NVIDIA) devices provide additional fast links between GPUs
within a node. These bypass the need to transfer data via the host.

.center[![:scale_img 30%](gpu-p2p.svg)]

This is referred to as **peer access**.



---
# Querying capability

In general, one should first check that peer access is available.

```cpp
hipDeviceCanAccessPeer(int *canAccessPeer, int device1, int device2);
```

The destination device is `device1` and the source device is `device2`.

It is also possible to disable and enable peer access.

```cpp
hipDeviceDisablePeerAccess(int peerDevice);
hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
```

The `flags` parameter is always set to zero.



---
# <span style="color:red">Exercise:</span> device-to-device data transfer

Write a simple program which allocates a large array (at least 10 MB)
on each of two devices using `hipMalloc()`.

By making repeated copies of the array with `hipMemcpy()`, measure the bandwidth which can be obtained
in the following scenarios.

1. copying from host to device and then from device to host;
2. copying directly from one device to another using `hipMemcpyDeviceToDevice` with peer access *disabled*;
3. repeating (2) with peer access enabled.

Note, we will need to adjust our queue submission script to ensure that two GPUs are available to the program.



---
# <span style="color:red">Next Lecture</span>

<br>
## [MPI](../13-mpi)