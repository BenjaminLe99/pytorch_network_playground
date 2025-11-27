# standard imports
import torch

# project imports
from models import create_model
from utils.logger import get_logger, TensorboardLogger
from data.cache import hash_config
from train_config import (
    config, dataset_config, model_building_config, optimizer_config, target_map,
)

logger = get_logger(__name__)

DEVICE = torch.device("cpu")
VERBOSE = False

logger.info(f"Running DEVICE is {DEVICE}")
# load data
tboard_writer = TensorboardLogger(name=hash_config(config))
logger.warning(f"Tensorboard logs are stored in {tboard_writer.path}")

# get weighted mean and std of expected batch composition
model_building_config["mean"] = torch.zeros(len(dataset_config["continous_features"]))
model_building_config["std"] = torch.ones(len(dataset_config["continous_features"]))

### Model setup
model = create_model.BNetDenseNet(dataset_config["continous_features"], dataset_config["categorical_features"], config=model_building_config)
model = model.to(DEVICE)
model = model.eval()
## load mean from marcel
import marcel_weight_translation as mwt
model = mwt.load_marcels_weights(model, continous_features=dataset_config["continous_features"])

cont_2 = torch.tensor([49.0, 16.9, 439.0, 68.0, 438.0, -27.3, -11.4, -33.9, 45.0, 39.6, 11.4, -2.7, 41.3, -39.6, -41.9, -1.6, 58.1, 1.0, 0.0, 0.9, 1.0, 96.6, 30.1, 101.9, 143.9, 0.9, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 86.3, 12.3, -0.0, -36.6, 202.0, 57.0, -11.8, 100.3, 288.3, 69.3, -11.8, 63.7, 0.0, 0.0, 0.0, 0.0]).reshape((1,-1))
cat_2 = torch.tensor([0, -1, 0, -1, 1, 1, 0]).reshape((1,-1))

# pred = model(cat, cont)
import export as ex
ex.torch_export(model, "test.pt2", (cat_2,cont_2))

from IPython import embed; embed(header="string - 60 in dnn_debug.py ")
