import logging

def setup_logging() -> None:
    # Set up basic configuration for the logging system
    # This configures the logging to display messages at the INFO level or higher
    logging.basicConfig(level=logging.INFO)
