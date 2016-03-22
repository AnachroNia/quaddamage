#What is added by me
--------------------
**All changes are in branch "Implicit-Surface"**
[in src/mesh.cpp](src/mesh.cpp)
```c++
double Mesh::binnedSAH(BBox bbox, const vector<int>& triangleList,const Axis &axis, int binCount);
double Mesh::SAH(BBox bbox, const vector<int>& triangleList, const Axis &axis);
double Mesh::calcCost(BBox bbox, double splitPos,const vector<int> &triangleList, const Axis &axis);
inline double calcArea(Vector A, Vector B);
```
Modified BuildKD to split on it's longest axis and to use a combination of 'SAH' and 'binned SAH'

[src/Expression.cpp](src/Expression.cpp) / [src/Expression.h](src/Expression.h)
A simple class that reads a math expression from string and serialize it in arrays for processing by [Evaluate.cu](src/Evaluate.cu)

[src/Evaluate.cu](src/Evaluate.cu) / [src/Evaluate.h](src/Evaluate.h)
A CUDA kernel that takes it's input from the Expression class and uses the Shunting yard algorithm to Evaluate the values of a function in points of a grid

[src/implicit_surface.cu](src/implicit_surface.cu) / [src/implicit_surface.h](src/implicit_surface.h) 
An implementation of the Marching Cubes algorithm - it uses Evaluate to compute the values of the function in a grid and uses these values to build a mesh

Note: To compile the project you need NVIDIA GPU Computing Toolkit v6.5

You can see some demos in [implicit_surfaces.md](implicit_surfaces.md) - including images, times to generate mesh and kd tree and render time (Tested on Geforce GT640M and i7-3630QM (Turbo boost disabled - 2,4Ghz))

# quaddamage

Another C++ raytracer for the v4 FMI raytracing course.
The course site is [http://raytracing-bg.net/](http://raytracing-bg.net/)

How to set it up on your machine
--------------------------------

On Windows:
-----------
   run scripts/download_sdk.py (requires Python 2.7, a prebuilt exe can be downloaded from [here](http://raytracing-bg.net/lib/download_sdk.exe)) to download and set up the SDKs and follow the instructions.
   This will install SDL and OpenEXR in a SDK/ subdirectory of the project, copy SDL.dll to this directory, and copy the project files from the versioned templates to a local, untracked copy.

On Linux and Mac OS X:
----------------------
   run scripts/downloda_sdk.py, and follow the instructions.


----------------------
