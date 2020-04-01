# -*- coding: utf-8 -*-

import logging

# TODO possible next steps
# TODO 1. (tobias) remove logging calls with \n und passe die logging aufrufe an (grade alle info)
# TODO 2. (pr) make cli flags to control verbosity (ich weiß grade nicht wie das gehen soll)
# TODO 3. (pr) be able to create a file config

path_to_logfile = 'spotpy.log'

# Standard defintion of main logger
spotpy_logger = logging.getLogger("spotpy")

# Add logging to stdout
handler_stdout = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler_stdout.setFormatter(formatter)
spotpy_logger.addHandler(handler_stdout)

# Add logging to file
handler_file = logging.FileHandler(path_to_logfile) # TODO this should be bound to cli arguments
handler_file.setFormatter(formatter)
spotpy_logger.addHandler(handler_file)

spotpy_logger.setLevel(logging.INFO) # TODO this should be bound to cli arguments
spotpy_logger.info('Write logging output to file \'%s\'', path_to_logfile)


def get_logger(name):
    """ Returns a new child logger for the main spotpy application logging

    Use this logger to return new childs of the main logger"""
    return spotpy_logger.getChild(name) # Set the logger name, with the implementation class
