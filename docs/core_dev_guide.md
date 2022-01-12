# ETA Core Developer's Guide

This document describes best practices for contributing to the core ETA
infrastructure. See `modules_dev_guide.md` to learn how to contribute modules
to ETA, which are more general and may even live outside of this codebase.


## Repository structure

We use the [GitFlow branching model](
https://datasift.github.io/gitflow/IntroducingGitFlow.html) for ETA.
Thus our repositories have protected `master` and `develop` branches, a
temporary *release branch* when we are ready to deploy a new release, and
multiple unprotected *feature branches* for work on new features. You can read
more about branching workflows in general [here](
https://git-scm.com/book/en/v2/Git-Branching-Branching-Workflows).

The `master` branch is the latest stable release of ETA. It is protected, and
it is only merged from a release branch. Each merge corresponds to a new ETA
release and is tagged with a version number. The one exception to this rule
are *hotfix branches*, which are directly merged into `master` if an emergency
bug fix is required.

The `develop` branch is the bleeding edge (mostly) stable version of ETA. It
is also protected and hence directly committing to it is not allowed.
Instead, when a feature is ready to be integrated, we open a pull request on
`develop`, which initiates a code chat (we prefer "code chat" to "code review",
since this should be a friendly endeavor!) where we discuss the changes and
ultimately merge them into `develop`.

A *release branch* is created from `develop` when we are ready to make a new
release. Only bugfixes (not new features) are committed to a release branch.
When the release is ready, the branch is merged into `master` (and back into
`develop`) and then deleted. The release is done!

*Feature branches* are where most of the development work is done. They are
unprotected, collaborative spaces to develop new features. When a new feature
is ready for deployment, a pull request is made to `develop`.


## Development workflow

Your typical workflow when contributing a new feature to ETA will be:

```shell
git checkout develop
git checkout -b <new_feature_branch>
git push -u origin <new_feature_branch>
# WORK
pre-commit run --files <changed_files>
# ADDRESS LINT OUTPUT
git status -s
git add <changed_files>
# COMMIT, PRE-COMMIT HOOKS (BLACK, PYLINT, ETC.) ARE RUN
git commit -m "message describing your changes"
# MORE WORK, LINTING, AND COMMITS
# PULL REQUEST
# CODE CHAT AND DISCUSSION
# MORE WORK, LINTING, AND COMMITS
# PULL REQUEST APPROVED AND MERGED
git branch -d <new_feature_branch>
```

Note that it is best practice to commit *often* in small, logical chunks rather
than combining multiple changes into a single commit.


## Python 2 and 3 compatibility

The ETA codebase is Python 2/3 cross-compatible. See the [Python 2/3
guide](https://github.com/voxel51/eta/blob/develop/docs/python23_guide.md) for
more tips on writing cross-compatible code in ETA.


## Style guide

We require all ETA code to adhere to our Python style guide. See the
[Python style guide](https://github.com/voxel51/eta/blob/develop/docs/python_style_guide.md)
for a description of our style, and see the
[linting guide](https://github.com/voxel51/eta/blob/develop/docs/linting_guide.md)
for details on our code linting tools.


## Image and video color formats

ETA uses an RGB format for images and video frames.  All functions that
read/write images and process images hence expect an RGB format.  This is not
always the case: the popular OpenCV library (on which ETA partially relies)
uses a BGR format.  This has two impacts:

- When we use OpenCV for IO, we explicitly convert to BGR from RGB, which
    although trivial can have a performance impact

- If you use OpenCV inside of your code that uses ETA for image and video
    frame representation, then you need to be careful to convert to BGR before
    actually invoking the `cv2` functions.  We provide ample conversion
    routines in the `eta.core.image` module


## Copyright

Copyright 2017-2022, Voxel51, Inc.<br>
voxel51.com
