from importlib import import_module


def __dir__():
    """
    Using the __dir__ and __getattr__ functions allows
    to inspect the availability of modules without loading them
    :return:
    """
    import pkgutil
    names = [
        name for importer, name, ispkg
        in pkgutil.iter_modules(__path__)
        if not ispkg and name != 'base'
    ]
    return names + ['custom', 'noData']


def __getattr__(name):
    names = __dir__()
    print(names)
    if name in names:
        try:
            db_module = import_module('.' + name, __name__)
        except ImportError:
            db_module = import_module('.base', __name__)
        return getattr(db_module, name)
    else:
        raise AttributeError('{} is not a member of spotpy.database')


def get_datawriter(dbformat, *args, **kwargs):
    """Given a dbformat (ram, csv, sql, noData, etc), return the constructor
        of the appropriate class from this file.
    """
    db_class = __getattr__(dbformat)
    datawriter = db_class(*args, **kwargs)
    return datawriter
