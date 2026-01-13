import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("boto3").setLevel(logging.CRITICAL)
logging.getLogger("botocore").setLevel(logging.CRITICAL)
logging.getLogger("uvicorn").setLevel(logging.ERROR)
logging.getLogger("fastapi").setLevel(logging.CRITICAL)
logging.getLogger("urllib3.connectionpool").setLevel(logging.CRITICAL)
logging.getLogger("celery").setLevel(logging.CRITICAL)
_logger = logging.getLogger(__name__)
_logger.propagate = True
