import logging


# This module sets up logging for the video processing application.
def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Print to console
        ]
    )
    return  logging.getLogger(__name__)