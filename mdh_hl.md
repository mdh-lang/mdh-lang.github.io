---
layout: single
---

# MDH High-Level Program Representation

The [MDH's High-Level Program Representation](../assets/files/publications/toplas24/paper.pdf) expresses data-parallel computations using exactly three higher-order functions:

- `inp_view` captures the accesses to the *input data*
- `md_hom`   expresses the *data-parallel computation*
- `out_view` captures the accesses to the *output data*

These three functions are always composed in the same order -- for expressing any data-parallel computation -- and do not require complex function nesting.

![MatVec Expressed in MDH's High-Level Program Representation](/assets/images/hl_matvec.png)

For the example of *Matrix-Vector Multiplication (MatVec)*, shown above, function `inp_view` specifies that MatVec has two inputs, matrix `M` of size `IxK` and vector `v` of size `K`, and at each point `(i,k)`, the matrix is accessed at position `(i,k)` and the vector at position `(k)`. Function `md_hom` expresses that matrix and vector elements are multiplied via `*` and that the obtained intermediate results are combined in the first dimension (a.k.a. `i`-dimension in the context of MatVec) via concatenation (denoted as `++` in MDH) and in the second dimension (a.k.a. `k`-dimension) via point-wise addition `+`. Function `out_view` is trivial in this example, but it can be used in more-advanced computations, e.g., *Matrix Multiplication (MatMul)*, to store the result matrix as transposed, etc.
