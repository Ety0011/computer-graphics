# TIMINGS TESTS WITH VARIOUS OPTIMIZATIONS

This file documents the timings with different optimizations over the vanilla code.
Use this with the python script for plotting (WIP)

### VANILLA

File: bunny_small.obj
Description: Vanilla Code, no modifications to the algorithms
Average Time Total: 82.58556
Average Time FPS: 0.01210865

### CONSTS_ONLY

File: bunny_small.obj
Description: glm was changed for raym, a constexpr-only library for vectors and matrices
Average Time Total: 41.523450000000004
Average Time FPS: 0.02408291
