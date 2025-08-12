template: titleslide
# Conjugate Gradient Solver



---
# Conjugate gradient: definition

We will now tackle a more substantial programming problem.

The conjugate gradient method provides a general method for the solution of linear systems (`Ax = b`) for symmetric matrices.

Click the link below for an explanation of the algorithm.

[https://en.wikipedia.org/wiki/Conjugate_gradient_method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)

It is not necessary to understand the details, as we are just interested in implementing the algorithm.



---
# Conjugate gradient: implementation

The first step performs a matrix-vector multiplication, see [solution_dgemv.hip.cpp](../../exercises/14-conjugate-gradient-solver).

The second computes a scalar residual from a vector residual, e.g. the residual for a vector "`r`" of length "`n`"
can be computed in serial like so.

```cpp
residual = 0.0;
for (int i = 0; i < n; i++) {
    residual += r[i]*r[i];
}
```

You should recognise this as the solution to an earlier exercise ([solution_ddot.hip.cpp](../../exercises/14-conjugate-gradient-solver)).

This exercise then involves using those two parts to construct the entire conjugate gradient algorithm.
Restrict yourself to the use of a single GPU.

Remember that you can always check your answer by multiplying out the solution to see if you recover the original right-hand side.
This can be done on the host.