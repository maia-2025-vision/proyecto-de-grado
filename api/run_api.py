import os

import uvicorn
from loguru import logger

from api.config import DEFAULT_SETTINGS

# If UVICORN_PORT, UVICORN_RELOAD env var  set then take values from them,
# otherwise take them from default settings
HOST = os.getenv("UVICORN_HOST", DEFAULT_SETTINGS.HOST)
PORT = int(os.getenv("UVICORN_PORT", DEFAULT_SETTINGS.PORT))
RELOAD = bool(os.getenv("UVICORN_RELOAD", DEFAULT_SETTINGS.RELOAD))


if __name__ == "__main__":
    logger.info(f"Starting uvicorn: HOST={HOST} PORT={PORT} RELOAD={RELOAD}")
    uvicorn.run("api.main:app", host=HOST, port=PORT, reload=RELOAD)
