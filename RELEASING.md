# Releasing the ETA package

> [!NOTE]
> These steps are to be performed by authorized Voxel51 engineers.

This repository is trunk based with the `main` branch as the trunk.
Every PR merges to `main` and nothing originates on a release branch.
Between releases, the `VERSION` file is the next planned version.
Reviewers of version-bump PRs should always check that the version
matches the tag being cut.

## Minor / major release (vX.Y.0)

1. Confirm the `VERSION` file on `main` is `X.Y.0`.

1. Navigate to the
   [releases page](https://github.com/voxel51/eta/releases)
   and select *Draft a new release*
1. From the *Tag: Select Tag* drop down,
   select *Create new tag*,
   enter `vX.Y.0`,
   select *Create*
1. Set the *Target* branch to `main`
1. Select *Generate release notes*
1. For the *Release label, select *Latest*
1. Select *Publish release*
1. Monitor the
   [publish workflow](https://github.com/voxel51/eta/blob/main/.github/workflows/publish.yml)
   (triggered by the tag push)
   and re-run until successful

   > [!NOTE]
   > The worklow builds the `.whl` artifacts and publishes them to
   > [PyPI](https://pypi.org/project/voxel51-eta/)

1. Validate latest version is published to
   [https://pypi.org/project/voxel51-eta/](https://pypi.org/project/voxel51-eta/)

1. Open a version-bump PR to `main` advancing `VERSION`
   to the next planned version

## Patch release (vX.Y.Z)

1. Merge the fixes on `main`
1. Create a new branch `release/vX.Y.Z` from the `vX.Y.Z-1` tag

    ```shell
    git checkout vX.Y.Z-1
    git checkout -b `release/vX.Y.Z`
    git push
    ```

1. Cherry pick the changes to the release branch via PR

    ```shell
    git checkout -b 'cherry-pick/fix-...'
    git cherry-pick -x <COMMIT_SHA>
    git push
    ```

    > [!NOTE]
    > A release branch accepts only cherry-picks of
    > `main` commits and its version bump.

1. Open a PR to the release branch bumping `VERSION` to `X.Y.Z`.
1. Navigate to the
   [releases page](https://github.com/voxel51/eta/releases)
   and select *Draft a new release*
1. From the *Tag: Select Tag* drop down,
   select *Create new tag*,
   enter `vX.Y.Z`,
   select *Create*
1. Set the *Target* branch to `release/vX.Y.Z`
1. Select *Generate release notes*
1. For the *Release label, select *Latest*
1. Select *Publish release*
1. Monitor the
   [publish workflow](https://github.com/voxel51/eta/blob/main/.github/workflows/publish.yml)
   (triggered by the tag push)
   and re-run until successful

    > [!NOTE]
    > The worklow builds the `.whl` artifacts and publishes them to
    > [PyPI](https://pypi.org/project/voxel51-eta/)

1. Validate latest version is published to
   [https://pypi.org/project/voxel51-eta/](https://pypi.org/project/voxel51-eta/)

## Release candidates

Create and push a tag `vX.Y.Z-rc.N` on the branch being released.
The publish workflow checks that the tag extends the `VERSION` file and
builds the rc version from the tag.
