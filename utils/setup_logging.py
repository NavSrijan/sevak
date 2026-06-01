import logging 
import colorlog
import sys

def setup_logging():
    color_log_format = "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    plain_log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%d-%m-%Y %H:%M:%S"

    color_handler = colorlog.StreamHandler(sys.stdout)
    color_handler.setFormatter(colorlog.ColoredFormatter(
        color_log_format,
        datefmt=date_format,
        log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'red,bg_white',
            }

        ))

    file_handler = logging.FileHandler("logs/app.log", mode="a")
    file_handler.setFormatter(logging.Formatter(plain_log_format, datefmt=date_format))

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            color_handler,
            file_handler
        ],
    )

    logging.getLogger("telegram").setLevel(logging.WARNING) 
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)

