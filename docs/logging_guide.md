# ETA Logging Guide

ETA uses the standard Python `logging` module to log messages. This document
describes how to use ETA's logging infrastructure.


## Logging in a module

By convention, ETA names loggers at the module-level, which propagate their
messages up to the root logger, which is configured to write logs to stdout
and/or log files.

Therefore, logging in an ETA module is as simple as:

```python
import logging

logger = logging.getLogger(__name__)

# A debugging message. Only recorded if logging is specifically customized to
# log messages at level logging.DEBUG and above
logger.debug(...)

# A standard info message
logger.info(...)

# A non-fatal warning message
logger.warning(...)

# There is no need to call logger.error(...) because ETA automatically
# captures raised exceptions and logs them before the process is terminated
raise Exception("A FATAL ERROR HERE")
```


## Logging configuration

ETA supports various mechanisms for customizing logging.

#### Default logging

By default, the top-level `eta/__init__.py` automatically configures logging
to write messages to stdout at level `logging.INFO` or greater the first time
any ETA module is imported:

```python
import eta

logger = logging.getLogger(__name__)

# Logging is printed to stdout
logger.info(...)
```

#### Basic logging

If you prefer ETA messages to be logged to stdout at a different level or
format, you can override the default configuration using the
`eta.core.log.basic_setup()` method. For example:

```python
import logging
from eta.core import log

logger = logging.getLogger(__name__)

log.basic_setup(level=logging.DEBUG, fmt="%(levelname)s:%(message)s")

# Logging is printed to stdout at the specified level and format
logger.debug(...)
```

#### Custom logging

You can perform more sophisticated custom logging configurations in ETA via
the `eta.core.log.custom_setup()` method. The configuration itself is defined
by instantiating the `eta.core.log.LoggingConfig` class.

For example, assuming `logging_config_path` contains a `LoggingConfig` JSON
dictionary, you can do:

```python
from eta.core import log

logger = logging.getLogger(__name__)

logging_config = log.LoggingConfig.from_json(logging_config_path)
log.custom_setup(logging_config)

# Logging is handled according to the `LoggingConfig` instance
logger.info(...)
```


## References

General logging resources

* https://docs.python.org/2/howto/logging.html

* https://docs.python.org/2.7/howto/logging-cookbook.html#logging-cookbook

* https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python

Formatter syntax

* https://docs.python.org/3.1/library/logging.html#formatter-objects


## Copyright

Copyright 2017-2022, Voxel51, Inc.<br>
voxel51.com
