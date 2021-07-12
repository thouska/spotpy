# -*- coding: utf-8 -*-
"""
Defines a standard logger for the main application, which every child can
  derive from. Also it's possible to use the logger for the main
  application.
"""

import logging
from datetime import datetime
from pathlib import PurePath

# Standard defintion of main logger
spotpy_logger = logging.getLogger("spotpy")

path_to_logfile = None
handler_stdout = None
handler_file = None

formatter_file = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
formatter_stdout = logging.Formatter('%(message)s')

def instantiate_logger(name, quiet=None, logfile=None, logdir=None):
    """
    Instantiate the logger,
      if already instantiated returns a new child of the main logger
    """

    path_to_logdir = PurePath() # current directory is assumed
    path_to_logfile = PurePath('{}-spotpy.log'.format(datetime.isoformat(datetime.now())))

    if not spotpy_logger.handlers:
      # create the handlers and call logger.addHandler(logging_handler)
      # Add logging to stdout
      handler_stdout = logging.StreamHandler()
      
      handler_stdout.setFormatter(formatter_stdout)
      spotpy_logger.addHandler(handler_stdout)

      if quiet:
        handler_stdout.setLevel(logging.ERROR)

      # Add logging to file
      if logdir is not None:
        path_to_logdir = logdir
      if logfile is not None:
        path_to_logfile = logfile
      path_to_logfile = PurePath(path_to_logdir, path_to_logfile)

      handler_file = logging.FileHandler(path_to_logfile)
      handler_file.setFormatter(formatter_file)
      spotpy_logger.addHandler(handler_file)

      spotpy_logger.setLevel(logging.INFO)  
      spotpy_logger.info('Write logging output to file \'%s\'', path_to_logfile)

    return get_logger(name)
    

def get_logger(name):
    """ Returns a new child logger for the main spotpy application logging.

    Use this logger to return new childs of the main logger"""
    
    return spotpy_logger.getChild(name)
