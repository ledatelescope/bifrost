# The Bifrost Tutorial

[![Build Status](https://fornax.phys.unm.edu/jenkins/buildStatus/icon?job=BifrostTutorial)](https://fornax.phys.unm.edu/jenkins/job/BifrostTutorial/)

A collection of examples that show how to use various features in the [Bifrost framework](https://github.com/ledatelescope/bifrost/).  Before beginning this tutorial it would be a good idea to famaliarize yourself with the framework and its concepts:

 * Bifrost is described in [Cranmer et al.](https://arxiv.org/abs/1708.00720)
 * Documentation for the Python APIs can be found [here](http://ledatelescope.github.io/bifrost/)
 * There is an [overview talk of Bifrost](https://www.youtube.com/watch?v=DXH89rOVVzg) from the 2019 CASPER workshop.
 
 ## Docker Image
 
 You should be able to run these tutorials in any Jupyter environment that has Bifrost installed.  We also
 have a Docker image with Bifrost, CUDA, Jupyter, and the tutorial already installed if you want a quicker
 path to trying it out.  Simply run:
 
 ```
 docker pull lwaproject/bifrost_tutorial
 docker run -p 8888:8888 --runtime=nvidia lwaproject/bifrost_tutorial
 ```
 
 This will start the Jupyter server on port 8888 which you can connect to with a browser running on the
 host.  *Note that this uses Nvidia runtime for Docker to allow access to the host's GPU for the GPU-enabled 
 portions of the tutorial.*
