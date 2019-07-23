# CUDA Installation Guide

This document will guide you through installing **CUDA 9.0** and **cuDNN v_7**.

It is highly recommended that you look at [this article](
https://www.tensorflow.org/install/source#tested_source_configurations), which
tells you which version of CUDA is compatible with each version of TensorFlow.

> WARNING: This is what worked for me and there could be issues pertraining to
> a specific system configuration. Nonetheless, I have tried to be as general
> as possible.


## Overview

There are three parts to this installation:

1. Installing the NVIDIA drivers
2. Installing CUDA
3. Installing cuDNN


## Installing the NVIDIA drivers

You can do this by running

```
sudo apt-get install nvidia-384 nvidia-modprobe
```

**Note**: You could try to install a newer version of the driver, but that
could give you compatibility issues depending on your CUDA version.

To verify the installation, run

```
nvidia-smi
```


## Installing CUDA

Get the latest CUDA runfile from the NVIDIA website that is compatible with the
version of TensorFlow you wish to use. As mentioned above, this guide is for
CUDA 9.0, which is compatible with TensorFlow 1.12.

Download the run file directly from the command line with

```
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
```

Once the run file is downloaded, do this:

```
chmod +x cuda_9.0.176_384.81_linux-run
./cuda_9.0.176_384.81_linux-run --extract=$HOME
```

The second command above extacts three files - one for NVIDIA driver
installation (which we don't need, because we have already installed that
above), one for CUDA installation and one for installing CUDA samples (which
we can use for verifying the CUDA installation, but it's optional).

Now navigate to your `HOME` directory and run the following:

```
sudo ./cuda-linux.9.0.176-22781540.run
sudo ./cuda-samples.9.0.176-22781540-linux.run
```

Now CUDA is installed but the system doesn't yet know where to find it. To
remedy that, add the following lines to your `~/.bashrc`:

```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Also, run

```
sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"
```

just in case. This is same as adding the runtime library to the
`LD_LIBRARY_PATH`.

**Fun fact**: whatever version of CUDA you install gets symlinked to
`/usr/local/cuda`. You can verify this by running `ls -l /usr/local/`. Thus
the above bash profile configuration works irrespective of the version of CUDA
you are installing.

To ensure that your bash profile changes take effect in your current shell,
you can run `source ~/.bashrc`.


### Verifying the installation

This is optional and takes time. However, this is wise to do in case there are
compatibility issues with your `gcc` installation, for example.

Run the following to test with the installed samples:

```
cd /usr/local/cuda/samples
sudo make
```

Note that this takes some time to run. See Extra Notes below if you get `gcc`
version errors here.

If the build is successful, you can run:

```
cd /usr/local/cuda/samples/bin/x86_64/linux/release
./deviceQuery
```


## cuDNN installation

As mentioned above, we will install version 7 of the cuDNN library. Go to the
[download page](https://developer.nvidia.com/rdp/cudnn-download)
and install the latest cuDNN 7* version made for CUDA 9.0.

Download all 3 .deb files: the runtime library, the developer library, and the
code samples library for your OS.

In your download folder, install them in the same order as downloaded
(first runtime, then dev, then docs). For example:

```
sudo dpkg -i libcudnn7_7.5.0.56-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.5.0.56-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.5.0.56-1+cuda10.1_amd64.deb
```

Now, add the following line to your `~/.bashrc`:

```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"
```


## FAQ

### GCC version error

In my case, the compiler threw a `gcc` version 6 needed error when I ran
`sudo make` from the `/usr/local/cuda/samples` folder. This happened because
I had `gcc` version 7 installed. I downgraded by doing the following:

```
sudo apt-get remove gcc-7
sudo apt-get remove g++-7

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
```

In some cases, before setting the alternatives, you might have to install
`gcc-6` and `g++-6`. It can be done by simply running:

```
sudo apt install gcc-6
sudo apt install g++-6
```

### `cudnn` or `cudart` not found

Make sure you have added the CUDA paths to `PATH` and `LD_LIBRARY_PATH` as
mentioned above.


## References

https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e.


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Yash Bhalgat, yash@voxel51.com
