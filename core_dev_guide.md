# ETA Core Developer's Guide

This document describes best practices for contributing to the ETA 
infrastructure.  Note that a separate document `modules_dev_guide.md` exists to 
specifically describe how to contribute modules to the ETA infrastructure, 
which can be outside of this codebase and repository.


## Repository structure and best practices

`master` branch is protected and cannot be directly merged.

`develop` branch is currently not protected.  However, we discourage direct commits to `develop` and rather encourage the use of a basic branching working for git.  Read more about it here: <https://git-scm.com/book/en/v2/Git-Branching-Branching-Workflows>

## Style Guide

Our codebase is Python.  Spacing is critical in Python.  First style rule: no tabs; only spaces;  Second style rule: 4 spaces.

Generally, we follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html?showone=Comments#Comments)

