import uvicorn
from loguru import logger

from api.config import SETTINGS

if __name__ == "__main__":
    logger.info(
        f"Starting uvicorn: HOST={SETTINGS.host} PORT={SETTINGS.port} RELOAD={SETTINGS.reload}"
    )
    uvicorn.run("api.main:app", host=SETTINGS.host, port=SETTINGS.port, reload=SETTINGS.reload)
