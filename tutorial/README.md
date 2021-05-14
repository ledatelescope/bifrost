# The Bifrost Tutorial

[![Build Status](https://fornax.phys.unm.edu/jenkins/buildStatus/icon?job=BifrostTutorial)](https://fornax.phys.unm.edu/jenkins/job/BifrostTutorial/)

A collection of examples that show how to use various features in the [Bifrost framework](https://github.com/ledatelescope/bifrost/).  Before beginning this tutorial it would be a good idea to familiarize yourself with the framework and its concepts:

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

## Amazon EC2 image

Another way to try the tutorials if you don't have appropriate GPU hardware is
using a virtual machine on Amazon's Elastic Compute Cloud (EC2). There are
GPU-enabled machine types that start at around US$0.53 per hour, which is pretty
affordable for playing around and learning. *Just don't forget to power down the
machine when not in active use.*

 1. Start at the [EC2 console for region
    us-west-1](https://us-west-1.console.aws.amazon.com/ec2/v2/home?region=us-west-1#Home:)
    and click the **Launch instance** button.

 2. Step 1: Choose AMI: switch to Community AMIs and paste
    `ami-0d13b7e8d7a746045` into the search bar. It should appear with the label
    `bifrost_tutorial`. Click **Select**.

 3. Step 2: Choose an Instance Type: filter by family **g4dn** then choose
    **g4dn.xlarge**. At the time of writing (May 2021), this instance was around
    US$0.53 per hour using on-demand pricing. Click **Next: Configure Instance**
    to proceed.

 4. Step 3: Configure Instance: On this screen, make sure **Auto-assign Public IP** is
    set to **Enable**. The rest of the defaults should be fine. Proceed to
    **Step 6**.

 5. Step 6: Configure Security Group. Create a **new security group** and make
    sure TCP ports 22 (SSH) is open to `0.0.0.0/0` (should be, by default).
    Then, **Add Rule** with **Custom TCP**, port **8888**, and source
    **Anywhere**. Click **Review and Launch**.

 6. Check the settings on the Review screen, especially **g4dn**, ports 22 and
    8888, and **Assign Public IP**. Click **Launch**.

 7. Next you will be prompted for an SSH key pair. If you've done this before
    (in us-west-1) and know you have access to the private key, you can choose
    that. We'll assume you **Create a new key pair**. Enter the key pair name
    `bifrost-gpu` and click **Download Key Pair**. Save the `bifrost-gpu.pem`
    file someplace safe, perhaps in `~/.ssh`. Now click **Launch instance(s)**.

 8. Back on your EC2 dashboard, it should (eventually) show it as Running, and
    list a Public IPv4 address. Make not of that address and substitute it for
    `MY_IP_ADDR` below.

 9. Make sure your downloaded private key file is protected, by running:

    ```
    chmod 600 ~/.ssh/bifrost-gpu.pem
    ```

 10. Log in to the new server's command line, like this:

     ```
     ssh -i ~/.ssh/bifrost-gpu.pem -l ubuntu MY_IP_ADDR
     ```

 11. After successful login, type

     ```
     ./launch_jupyter.sh
     ```

     This will create a new self-signed security certificate and run the Jupyter
     notebook server. It should print a URL with a hexadecimal token in the
     output.

 12. Open `https://MY_IP_ADDR:8000/` in your web browser. It will display a
     security warning due to the self-signed certificate. You can select
     **Advanced** and **Accept the risk** (exact prompts will depend on your
     browser).

 13. Next the page will prompt for a password or token. Copy and paste the token
     from the terminal output of the launch script. After authenticating, you
     should see folders including `bifrost_tutorial`.

 14. When finished, you can use control-C to and `y` in the terminal to stop the
     notebook server. Don't forget to shut down the EC2 instance using `sudo
     poweroff`. (Note that each time you shut down and restart the instance, it
     can have a new IP address.)
