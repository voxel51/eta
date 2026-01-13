# ETA Logging Guide

ETA uses the standard Python `logging` module to log messages. This document
describes how to use ETA's logging infrastructure.

## Logging in a module

By convention, all ETA library code uses module-level loggers:

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

If you are running [ETA modules](modules_dev_guide.md) or
[pipelines](pipelines_dev_guide.md), `eta.core.logging.custom_setup()` is
implicitly called during execution:

```py
import eta.core.logging as etal

# This is how pipelines configure logging
etal.custom_setup(pipeline_config.logging_config)

# This is how individual modules configure logging
etal.custom_setup(module_config.base.logging_config)
```

However, if you are simply using ETA as a dependency, no logging configuration
is automatically performed.

Note that, when invoked, ETA's logging configuration modifies your **root
logger**.

#### Basic logging

You can use the `eta.core.logging.basic_setup()` utility to quickly configure
your root logger to log to stdout at a specified level and/or with a specified
format:

```python
import logging
import eta.core.logging as etal

logger = logging.getLogger(__name__)

# Default: log messages only at INFO level or higher
etal.basic_setup()
logger.info("Hello, world!")

# Custom logging level/format
etal.basic_setup(level=logging.DEBUG, fmt="%(levelname)s:%(message)s")
logger.debug("Hello, world!")
```

#### Custom logging

You can use the `eta.core.logging.custom_setup()` method to perform more
sophisticated custom logging.

The configuration itself is defined by instantiating the
`eta.core.logging.LoggingConfig` class. For example, you can configure
dynamically:

```python
import logging
import eta.core.logging as etal

logger = logging.getLogger(__name__)

# Customize logging via `LoggingConfig` instance
logging_config = etal.LoggingConfig.default()
logging_config.stream_to_stdout = False
logging_config.filename = "/tmp/eta.log"
print(logging_config)

etal.custom_setup(logging_config)

logger.info("Hello, world!")
```

```shell
cat /tmp/eta.log
rm /tmp/eta.log
```

Or you can store logging configuration on disk:

```json
{
    "stream_to_stdout": false,
    "filename": "/tmp/eta.log"
}
```

and load it as follows:

```python
import eta.core.logging as etal

logging_config = etal.LoggingConfig.from_json("/path/to/logging_config.json")
print(logging_config)
```

## References

General logging resources

-   https://docs.python.org/2/howto/logging.html

-   https://docs.python.org/2.7/howto/logging-cookbook.html#logging-cookbook

-   https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python

Formatter syntax

-   https://docs.python.org/3.1/library/logging.html#formatter-objects

## Copyright

Copyright 2017-2026, Voxel51, Inc.<br> voxel51.com
