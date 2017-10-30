# ETA linting


## Installation

```shell
pip install pycodestyle
echo [pycodestyle] > ~/.config/pycodestyle

pip install pylint
pylint --generate-rcfile > ~/.pylintrc
```


## Linting a file

```shell
pycodestyle <file>
pylint <file>
```


## Customizing pylint

Edit `~/.pylintrc` and by setting the following values:

```
max-line-length=79
output-format=colorized
reports=no
score=no
```


## Customizing pycodestyle

Edit `~/.config/pycodestyle` and set the following values:

```shell
max-line-length=79
```

See the [user guide](https://pycodestyle.readthedocs.io/en/latest/intro.html)
for more information.
