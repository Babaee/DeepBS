How to generate background image from video sequence
-------------------------------------------------------
The code for generating background image is located in 
./litiv/samples/changedet/src

before editing the code, install the following dependencies
1. **[CMake](https://cmake.org/) >= 3.1.0 (required)**

2. **[OpenCV](http://opencv.org/) >= 3.0.0 (required)**
3. OpenGL >= 4.3 (optional, for GLSL impl)
4. [GLFW](http://www.glfw.org/) >= 3.0.0 or [FreeGLUT](http://freeglut.sourceforge.net/) >= 2.8.0 (optional, for GLSL implementations)
5. [GLEW](http://glew.sourceforge.net/) >= 1.9.0 (optional, for GLSL implementations)
6. [GLM](http://glm.g-truc.net/) (optional, for GLSL implementations)
7. (CUDA/OpenCL will eventually be added as optional)
8. (OpenGM + Gurobi/CPLEX/HDF5 will eventually be added as optional)

For more information of the dependencies, please chcek [Litiv](https://github.com/plstcharles/litiv)

Open the main.cpp in the above mentioned directory, edit the file path on line 144 and line 145
e.g. if the video sequence is stored in /usr/dataset/video/in000001.jpg ....
(Note that by default the video input should be renamed to in000001.jpg, in000002.jpg ....)
Then change ´fileName_prefix´ to "/usr/dataset/video/in", the frame number string will be generated automatically. Edit outputfileName_prefix to anywhere you want to store the generated background image.

To compile the code in ubuntu, first, in the command line cd in the directory ./litiv, make a directory called build, then cd in build, type cmake .. & make. The project should then be compiled. the binary file can be found in the subdirectory in build.

For windows user, first use cmake GUI to generate Microsoft Visual Studio project file, then in Visual Studio right click the project directory we are working on, choose build solution.


