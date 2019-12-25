import logging
from typing import Tuple, Any
import os
import numerox as nx

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s = %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def get_logger(name, level=logging.INFO) -> Any:
    """Returns logger object with given name"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger 


def prepare_tournament_data() -> Tuple:
    """Downloads latest the tournament data from numerox"""

    tournaments: List[str] = nx.tournament_names()
    data: nx.data.Data = nx.download('numerai_dataset.zip')
    
    return tournaments, data


