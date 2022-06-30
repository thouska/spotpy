from __future__ import division, print_function, unicode_literals

from . import algorithms, database, describe
import click
import inspect
import io
import os


def get_config_from_file():
    """
    Gets the spotpy configuration from a config file 'spotpy.conf'.

    Example:

        sampler = mc
        dbtype = csv
        parallel = seq
        # This is a comment
        runs = 10
    """
    config = {}
    if os.path.exists('spotpy.conf'):
        with io.open('spotpy.conf') as f:
            for line in f:
                if not line.strip().startswith('#'):
                    try:
                        k, v = line.split('=', 1)
                        config[k.strip()] = v.strip()
                    except ValueError:
                        pass
    return config


def get_sampler_from_string(sampler_name):
    return getattr(algorithms, sampler_name)


def make_type_from_module(module, *exclude):

    def use(cl):
        # Check if class name starts with an exclusion term
        return inspect.isclass(cl) and not any([cl.__name__.startswith(ex) for ex in ('_', ) + exclude])
    members = inspect.getmembers(module, use)
    return click.Choice([n for n, m in members if not n.startswith('_')])


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    pass


@cli.command()
@click.pass_context
@click.option('--sampler', '-s', type=make_type_from_module(algorithms), default='mc',
              help='Select the spotpy sampler')
@click.option('--dbformat', type=click.Choice(database.__dir__()), default='ram',
              help='The type of the database')
@click.option('--dbname', type=click.STRING, help='The name of the database, leave open for ram')
@click.option('--parallel', '-p', type=click.Choice(['seq', 'mpc', 'mpi']), default='seq',
              help='Parallelization: seq = no parallelization, mpi = MPI (for clusters), mpc = multiprocessing')
@click.option('--runs', '-n', type=click.INT, default=1, help='Number of runs')
@click.option('--config', '-c', is_flag=True,
              help='Print only the configuration, can be used to create a config file with your_model.py > spotpy.conf')
def run(ctx, **kwargs):
    """
    Runs a sampler for automatic calibration
    """
    setup = ctx.obj
    if kwargs.pop('config', None):
        click.echo('\n'.join('{} = {}'.format(k, v) for k, v in kwargs.items()))
    else:
        sampler_name = kwargs.pop('sampler')
        sampler_class = get_sampler_from_string(sampler_name)
        runs = kwargs.pop('runs')
        sampler = sampler_class(setup, **kwargs)
        sampler.sample(runs)


@cli.command()
@click.pass_context
def gui(ctx):
    """
    Shows a GUI for manual calibration
    """
    from spotpy.gui.mpl import GUI
    setup = ctx.obj
    gui = GUI(setup)
    gui.show()


def main(setup):
    # Prevent help text from wrapping
    cli.help = '\b\n' + describe.setup(setup).replace('\n\n', '\n\b\n')
    config = get_config_from_file()
    cli(obj=setup, auto_envvar_prefix='SPOTPY', default_map=config)

