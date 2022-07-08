# The Bifrost Tutorial

A collection of examples that show how to use various features in the [Bifrost framework](https://github.com/ledatelescope/bifrost/).  Before beginning this tutorial it would be a good idea to familiarize yourself with the framework and its concepts:

 * Bifrost is described in [Cranmer et al.](https://arxiv.org/abs/1708.00720)
 * Documentation for the Python APIs can be found [here](http://ledatelescope.github.io/bifrost/)
 * There is an [overview talk of Bifrost](https://www.youtube.com/watch?v=DXH89rOVVzg) from the 2019 CASPER workshop.
 * There is a [walk through of this tutorial](https://youtu.be/ktk2dkUssAA?t=20170) from the 2021 CASPER workshop.

You should be able to run these notebooks in any Jupyter environment that has Bifrost installed — just open the `.ipynb` files in this directory.  To try them without installing Bifrost or configuring Jupyter locally, you can open them in Google Colab (free, cloud) or use Docker (local, requires GPU hardware but dependencies are bundled).

## Google Colab

This is the simplest way to try the tutorial, without needing to install or configure anything. It also does not require a local GPU: Google currently provides free access to one GPU-enabled runtime (for foreground computation only).

Visit each `.ipynb` file in this directory on GitHub, where you can read the text, code, and see static output.  To allow interaction, use the “Open in Colab” button at the top.  Then use control-enter to run each selected code block; the first one should install Bifrost on your runtime instance. (It can take a short while, but a green arrow should march through the steps. If you switch away, just remember to return before the instance times out!)

Please [report any issues](https://github.com/ledatelescope/bifrost/issues) with opening, installing, or running the tutorial notebooks on Colab.

## Docker Image

We provide a Docker image with Bifrost, CUDA, Jupyter, and the tutorial already installed if you want a quicker path to trying it out.  Simply run:

 ```
 docker pull lwaproject/bifrost_tutorial
 docker run -p 8888:8888 --runtime=nvidia lwaproject/bifrost_tutorial
 ```

 This will start the Jupyter server on port 8888 which you can connect to with a browser running on the
 host.  *Note that this uses Nvidia runtime for Docker to allow access to the host's GPU for the GPU-enabled
 portions of the tutorial.*
