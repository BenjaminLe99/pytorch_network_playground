import hashlib
import os
import pathlib
import pickle

from utils.logger import get_logger

logger = get_logger(__name__)

class DataCacher():
    def __init__(self, config):
        self.path = self.cache_dir(config)

    def cache_dir(self, config):
        h = tuple(config.items())
        h = hashlib.sha256(str(h).encode("utf-8")).hexdigest()[:10]
        p = pathlib.Path(os.environ["CACHE_DIR"])
        
        if not p.exists():
            raise FileExistsError("Cache dir does not exist")
        return p / h

    def create_cache_dir(self):
        self.path.mkdir(parents=False, exist=False)

    def save_cache(self, data):
        logger.info(f"Saving cache at {self.path}:")
        with open(f"{self.path}", "wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Done saving cache in {self.path}")

    def load_cache(self):
        if not self.path.exists():
            raise FileExistsError(f"Cache path {self.path} does not exist")

        logger.info(f"Loading cache from {self.path}")
        with open(self.path, "rb") as file:
            events = pickle.load(file)
        logger.info(f"Done loading cache from {self.path}")
        return events
