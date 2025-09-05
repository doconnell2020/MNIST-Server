import os

if os.getenv("ENVIRONMENT") is None:
    from .local import config
