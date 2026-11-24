"""Runs a module of src/ as __main__ with given arguments, so that its parallel workers find its functions; used by
bstd_return_70.py."""
import runpy
import sys


def run(module, *args):
    """Run `module` (a module name inside src/) as if started with ``python -m module args``.
    Arguments given on the command line of the entry script are appended."""
    argv = sys.argv
    sys.argv = [module] + [str(a) for a in args] + argv[1:]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:              # a module that calls sys.exit(0) has finished normally
        if e.code not in (None, 0):
            raise
    finally:
        sys.argv = argv
