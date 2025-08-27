import logging

from rich.logging import RichHandler


class LogOncePerLocationFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.logged_locations = set()

    def filter(self, record) -> bool:
        key = (record.pathname, record.lineno)
        if key in self.logged_locations:
            return False
        self.logged_locations.add(key)
        return True


def setup_logging(level=logging.INFO, path=None, record=True, once=False) -> None:
    if not record:
        return

    handler = RichHandler()

    if once:
        handler.addFilter(LogOncePerLocationFilter())

    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]", handlers=[handler])
    logger = logging.getLogger()
    logger.propagate = False
