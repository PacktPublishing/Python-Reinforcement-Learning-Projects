import subprocess
import sys
from utils import timer

import os

from constants import PATHS

import logging

logger = logging.getLogger(__name__)

def main():

    if not os.path.exists(PATHS.SELFPLAY_DIR):
        with timer("Initialize"):
            logger.info('==========================================')
            logger.info("============ Initializing...==============")
            logger.info('==========================================')
            res = subprocess.call("python controller.py initialize-random-model", shell=True)

        with timer('Initial Selfplay'):
            logger.info('=======================================')
            logger.info('============ Selplaying...=============')
            logger.info('=======================================')
            subprocess.call('python controller.py selfplay', shell=True)

    while True:
        with timer("Aggregate"):
            logger.info('=========================================')
            logger.info("============ Aggregating...==============")
            logger.info('=========================================')
            res = subprocess.call("python controller.py aggregate", shell=True)
            if res != 0:
                logger.info("Failed to gather")
                sys.exit(1)

        with timer("Train"):
            logger.info('=======================================')
            logger.info("============ Training...===============")
            logger.info('=======================================')
            subprocess.call("python controller.py train", shell=True)

        with timer('Selfplay'):
            logger.info('=======================================')
            logger.info('============ Selplaying...=============')
            logger.info('=======================================')
            subprocess.call('python controller.py selfplay', shell=True)

        with timer("Validate"):
            logger.info('=======================================')
            logger.info("============ Validating...=============")
            logger.info('=======================================')
            subprocess.call("python controller.py validate", shell=True)


if __name__ == '__main__':
    main()
