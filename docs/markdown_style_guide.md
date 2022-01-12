# ETA Markdown Style Guide

We require all Markdown files contributed to ETA to adhere to our style. Our
priority is _consistency_, so that developers can quickly digest the guide and
not be distracted by strange formatting. _When in doubt, follow the existing
style in the file you are contributing to._

We use Prettier (deterministic) auto-formatting for all Markdown files as a
pre-commit hook. Installing ETA with the `-d` (dev) flag automatically
configures these hooks. Much of the style guide is automatically handled by
Prettier. See the
[linting guide](https://github.com/voxel51/eta/blob/develop/docs/linting_guide.md)
for more information.

All Markdown must obey the
[GitHub Markdown Syntax](https://guides.github.com/features/mastering-markdown).
It is highly recommended that you install the
[grip](https://github.com/joeyespo/grip) package, which lets you render your
Markdown files locally on your machine using GitHub's API to ensure that the
file will render correctly at https://github.com.

You can render a Markdown file with `grip` by running:

```
grip -b --user=<github-user> --pass=<api-key> /path/to/markdown/file.md
```

All Markdown in ETA must obey the following rules:

-   Maximum line length is **79 characters**, with the exception of long URLs
    that cannot be split

-   All Markdown files start with a title with `# Uppercase Title Words` syntax

-   Section headings use `## Capital then lowercase` syntax

-   Leave exactly one blank line before all of the following:

    -   section headings `##`
    -   lower level headings `###` and `####`
    -   paragraphs, before and after code blocks, and between top-level list
        items

-   One blank line at the end of the file

-   Indent 4 spaces when writing multiline list items

-   All terminal commands that you expect readers to run should user the triple
    backtick code block syntax:

```
bash /your/command/here
```

-   Use `single backticks` liberally to render variables, constants, or other
    important concepts in your text

-   All Markdown files must include the following copyright block at the
    bottom:

```
## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
```

## Copyright

Copyright 2017-2022, Voxel51, Inc.<br> voxel51.com
