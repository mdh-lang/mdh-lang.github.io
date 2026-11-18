---
layout: single
---

# MDH Code-Level Program Representation

MDH's Code-Level Program Representation (`CL-IR`) expresses data-parallel computations in an imperative, C-like form with loops, control flow, and memory allocation.
It is deliberately restricted to data-parallel computations, which makes it considerably easier to transform into executable target code than a general-purpose intermediate representation.