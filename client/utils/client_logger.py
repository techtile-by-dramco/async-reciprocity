import logging
from datetime import datetime
from typing import Optional


class LogFormatter(logging.Formatter):
    """Custom log formatter that prints timestamps with fractional seconds."""

    @staticmethod
    def pp_now() -> str:
        """Return the current time of day as a formatted string with milliseconds."""
        now = datetime.now()
        return "{:%H:%M}:{:05.2f}".format(now, now.second + now.microsecond / 1e6)

    def formatTime(self, record, datefmt: Optional[str] = None) -> str:  # noqa: N802
        """Override the default time formatter to include fractional seconds."""
        converter = self.converter(record.created)
        if datefmt:
            return converter.strftime(datefmt)
        return LogFormatter.pp_now()


def get_logger(name: str = __name__, level: int = logging.DEBUG) -> logging.Logger:
    """Create or return a logger configured with the custom formatter."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler()
        formatter = LogFormatter(
            fmt="[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s"
        )
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


__all__ = ["LogFormatter", "get_logger"]
