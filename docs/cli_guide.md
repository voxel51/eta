# Command-Line Interface Guide

Installing ETA automatically installs `eta`, a command-line interface (CLI) for
interacting with the ETA Library. This utility provides access to many useful
features of ETA, including building and running pipelines, downloading models,
and interacting with remote storage.

This document provides an overview of using the CLI.


## Quickstart

To see the available top-level commands, type `eta --help`. You can learn more
about any available subcommand via `eta <command> --help`.

For example, to see your current ETA config, you can `eta config --print`.


## Tab completion

To enable tab completion in `bash`, add the following line to your `~/.bashrc`:

```shell
eval "$(register-python-argcomplete eta)"
```

To enable tab completion in `zsh`, add these lines to your `~/.zshrc`:

```shell
autoload bashcompinit
bashcompinit
eval "$(register-python-argcomplete eta)"
```

To enable tab completion in `tcsh`, add these lines to your `~/.tcshrc`:

```shell
eval `register-python-argcomplete --shell tcsh eta`
```


## Usage

The following usage information was generated via `eta --all-help`:

> Last generated on 2020/01/06

```

```


## Copyright

Copyright 2017-2019, Voxel51, Inc.<br>
voxel51.com

Brian Moore, brian@voxel51.com
