# -*- coding: utf-8 -*-
"""
Defines a standard logger for the main application, which every child can
  derive from. Also it's possible to use the logger for the main
  application.
"""

import logging

path_to_logfile = 'spotpy.log'

# Standard defintion of main logger
spotpy_logger = logging.getLogger("spotpy")
if not spotpy_logger.handlers:
    # create the handlers and call logger.addHandler(logging_handler)
    # Add logging to stdout
    handler_stdout = logging.StreamHandler()
    formatter_file = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s ' +
                                  '%(message)s')
    formatter_stdout = logging.Formatter('%(message)s')

    handler_stdout.setFormatter(formatter_stdout)
    spotpy_logger.addHandler(handler_stdout)
    
    # Add logging to file
    handler_file = logging.FileHandler(path_to_logfile)  # TODO this should be bound to cli arguments
    handler_file.setFormatter(formatter_file)
    spotpy_logger.addHandler(handler_file)
    
    spotpy_logger.setLevel(logging.INFO)  # TODO this should be bound to cli arguments
    spotpy_logger.info('Write logging output to file \'%s\'', path_to_logfile)
    

def get_logger(name):
    """ Returns a new child logger for the main spotpy application logging

    Use this logger to return new childs of the main logger"""
    # Set the logger name, with the implementation class
    return spotpy_logger.getChild(name)
