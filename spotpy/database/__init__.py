from importlib import import_module


def get_datawriter(dbformat, *args, **kwargs):
    """Given a dbformat (ram, csv, sql, noData, etc), return the constructor
        of the appropriate class from this file.
    """
    try:
        db_module = import_module('.' + dbformat, __name__)
    except ImportError:
        db_module = import_module('.base', __name__)
    db_class = getattr(db_module, dbformat)
    datawriter = db_class(*args, **kwargs)
    return datawriter
