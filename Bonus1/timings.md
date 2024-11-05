# TIMINGS TESTS WITH VARIOUS OPTIMIZATIONS

This file documents the timings with different optimizations over the vanilla code.
Use this with the python script for plotting (WIP)

### VANILLA

File: bunny_small bunny_small
Description: Vanilla Code, no modifications to the algorithms
Average Time Total: 161.6175
Average Time FPS: 0.0062
Total Number of Triangles: 2784
Time per Triangle: 0.05805226### CONSTS_ONLY

File: bunny_small bunny_small
Description: glm was changed for raym, a constexpr-only library for vectors and matrices
Average Time Total: 79.9661
Average Time FPS: 0.0125
Total Number of Triangles: 2784
Time per Triangle: 0.02872347