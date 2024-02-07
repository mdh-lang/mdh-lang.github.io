---
layout: single
---

# MDH Lowering: From High-Level to Low-Level

The [MDH's Lowering Process](/under_review) fully automatically lowers a data-parallel computation expressed in its high-level program representation to an architecture- and data-optimized instance in its low-level representation.
For this, we have designed MDH's formal lowering process as generic in performance-critical parameters, such as the size of tiles and number of threads -- different values of these parameters lead to semantically equal instances of our low-level representation, but the instances are differently optimized.
Currently, MDH uses the [Auto-Tuning Framework (ATF)](www.atf-tuner.org) to fully automatically determine architecture- and data-optimized values of tuning parameters.

Overview of *MDH Tuning Parameters:*

![MDH Tuning Parameter](/assets/images/tp_tabelle.png)