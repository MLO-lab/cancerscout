import logging
import logging.config

from pathlib import Path

init_path = Path(__file__).resolve()

logging.config.fileConfig(
    init_path.parent / "log.conf",
    disable_existing_loggers=False,
)
logging.getLogger().setLevel(logging.INFO)

__version__ = "0.1.0"
