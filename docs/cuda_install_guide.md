# CUDA Installation Instructions
### Compiled by Yash Bhalgat (yash@voxel51.com)
#### Warning: This is what worked for me and I have tried to be as general as possible. But if any of the steps does not work for you, please do not come to my house. I do not have food for you.

This document will guide you through installing **CUDA 9.0** and **cuDNN v_7**.

[HIGHLY RECOMMENDED] Have a look at this [link](https://www.tensorflow.org/install/source#tested_source_configurations) which tells you
what version of TensorFlow works with what version of CUDA.

The latest version of CUDA available at the time of writing is 10.1, but no tf version is compatible with that.
So, we stick with CUDA 9.0 in this document.

There are three parts to this installation - 
1. Installing the NVIDIA drivers
2. Installing CUDA
3. Installing cuDNN

## 1. Installing the NVIDIA drivers
You can do this by running `sudo apt-get install nvidia-384 nvidia-modprobe`.

**Note**: You can even install a newer version of the driver.
But that could give you compatibility issues depending on your CUDA version.

To verify the installation, run `nvidia-smi`.

## 2. Installing CUDA
Get the latest CUDA runfile from the NVIDIA website. As mentioned above, the latest version available at
the time of writing is 10.1, but we will stick with 9.0

Download the run file directly from the command line with 
```
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
```

Once the run file is downloaded, do this:
```
chmod +x cuda_9.0.176_384.81_linux-run
./cuda_9.0.176_384.81_linux-run --extract=$HOME
```
The second command above basically extacts three files - one for NVIDIA driver installation (which we don't need,
because we have already installed that above), one for CUDA installation and one for installing CUDA samples (which 
we can use for verifying the CUDA installation, but it's optional).

Now, go to the HOME folder and run the following:
```
sudo ./cuda-linux.9.0.176-22781540.run
sudo ./cuda-samples.9.0.176-22781540-linux.run
```
Note that these commands have to be run with sudo access. So, CUDA is installed but the system doesn't know yet
about that. We need to add stuff to some paths so that the system can find the installation.

Add the following lines to your `~/.bashrc`:
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Also, run `sudo bash -c "echo /usr/local/cuda/lib64/ > /etc/ld.so.conf.d/cuda.conf"`, just in case. This is same as adding 
the runtime library to the `LD_LIBRARY_PATH`.

**Fun Fact**: Whatever version of CUDA you are installing, it gets symlinked to `/usr/local/cuda`. You could check this by
running `ls -l /usr/local/` and you will see some arrows there. So, when you add it to the paths like above, you can just add
the paths from `/usr/local/cuda` irrespective of the version of CUDA you are installing.

Coolio! Now that you have added the lines to `~/.bashrc`, run `source ~/.bashrc` (or restart your system or terminal).

### 2.1 Verifying the installation
This is optional and takes time. but it would be useful if there are some compatibilities issues with your `gcc` 
installation, for example.

Run the following to test with the installed samples:
```
cd /usr/local/cuda/samples
sudo make
```
Note that this takes some time to run. See Extra Notes below if you get `gcc` version errors here.

If you pass this successfully, you can run these lines:
```
cd /usr/local/cuda/samples/bin/x86_64/linux/release
./deviceQuery
```

## 3. cuDNN installation
As mentioned above, we will be installing version 7 of the cuDNN library. Go to the [download page](https://developer.nvidia.com/rdp/cudnn-download)
and install the latest cuDNN 7* version made for CUDA 9.0.

Download all 3 .deb files: the runtime library, the developer library, and the code samples library for your OS.

In your download folder, install them in the same order as downloaded (first runtime, then dev, then docs). 
For me, the commands were:
```
sudo dpkg -i libcudnn7_7.5.0.56-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.5.0.56-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.5.0.56-1+cuda10.1_amd64.deb
```

Now, add the following line to your `~/.bashrc`:
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"
```

## 4. Extra Notes
### 4.1 GCC pains
In my case, the compiler threw a `gcc` version 6 needed error when I ran `sudo make` from the `/usr/local/cuda/samples` folder.
This happens because your default gcc compiler might be a higher version that what is required. To solve this, you could do the
following:
```
Because I had gcc version 7 installed, first removed it.
sudo apt-get remove gcc-7 g++-7

Then installed gcc version 7, if not already present.
sudo apt-get remove gcc-7 g++-7

And then update the alternatives
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
```

### 4.2 `cudnn` not found or `cudart` not found
Make sure you have added the cuda paths to PATH and LD_LIBRARY_PATH as mentioned in this doc.

## Sources
[1] This tutorial: [LINK](https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e)
