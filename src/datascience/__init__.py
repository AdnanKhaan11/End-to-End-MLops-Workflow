import os
import sys
import logging

dir = "log"
logging_file = os.path.join(dir, "logging.log")
os.makedirs(dir, exist_ok=True)

log_formate = "[%(asctime)s - %(levelname)s - %(module)s - %(message)s]"

logging.basicConfig(
    # filename=logging_file,
    format=log_formate,
    level=logging.INFO,
    handlers=[logging.FileHandler(logging_file), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("End-to-End-MLops-Workflow")
