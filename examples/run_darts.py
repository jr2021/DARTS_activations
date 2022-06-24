import logging
from naslib.defaults.trainer import Trainer
from naslib.optimizers import DARTSOptimizer, GDASOptimizer, RandomSearch
from naslib.utils import set_seed, setup_logger, get_config_from_args
from docs.simple_activation_space import DartsSearchSpace

config = get_config_from_args(config_type="nas")
set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

search_space = DartsSearchSpace() 

optimizer = DARTSOptimizer(config)
optimizer.adapt_search_space(search_space)

trainer = Trainer(optimizer, config)
trainer.search()
trainer.evaluate()

