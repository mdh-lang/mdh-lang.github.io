---
layout: single
---

# MDH Low-Level Program Representation

The [MDH's Low-Level Program Representation](/under_review) explicitly expresses a de- and re-composition of a data-parallel computation for the memory and core hierarchies of a parallel architecture.
Consequently, the low-level representation allows formally reasoning about optimizations, and it particularly allows straightforwardly generating executable program code from it (e.g., in OpenMP, CUDA, or OpenCL) as the major optimization decisions are already expressed in the representation.

![MatVec Expressed in MDH's Low-Level Program Representation](/assets/images/ll_matvec.png)

Above, we show a low-level program instance for the example *Matrix-Vector Multiplication (MatVec)*; for simplicity, we consider in this example a simple, artificial target architecture that consists of two memory layers `HM` (Host Memory) and `L1` (L1 Cache), and one core layer `COR` only.
The *de-composition phase* (right part of the figure) partitions the input, step by step, for each memory and core layer and in each of MatVec's two dimensions (called `i` and `k` dimension).
Afterwards, the *scalar phase* (bottom part of the figure) multiplies the partitioned matrix and vector elements.
Finally, the *re-composition phase* (left part of the figure) combines the intermediate results in `i`-dimension via concatenation and in `k` dimension via point-wise addition.


[//]: # (Visualization of a straightforward low-level instance for MatVec:)

[//]: # ()
[//]: # (TODO: alternativ/zusäztlich video visualizer ?)