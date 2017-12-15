# ETA Data Download Guide

This document describes how to download (and upload) data for ETA in the
`public` Voxel51 Team Drive.

> NOTE: ETA relies on the IDs of the files on drive.google.com, so they must
> remain constant!


## Downloading files from Google Drive

You can download a file from ETA using the `file-id` from the sharing link:

```
https://drive.google.com/a/voxel51.com/file/d/<file-id>
```

You can also download individual files using the ETA module `gdrive_download.py`.


## Single files

To add a new file:

* upload the file

* click share > turn on link sharing > allow external access

To modify an existing file:

* right-click > Manage versions... > UPLOAD NEW VERSION


## Directories

Since drive.google.com does not support programmatic downloads of entire
directories, we maintain zip files of the directories, which can then be
downloaded and unzipped.

To add a new directory:

* create a directory and add files to it

* download the directory, create a zip of it, and add the zip file using the
instructions for adding single files above

To modify an existing directory:

* add new files to the directory

* download the directory and create a new zip of it

* right-click on the existing zip > Manage versions... > UPLOAD NEW VERSION
