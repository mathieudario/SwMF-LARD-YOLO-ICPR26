
def skip(line, cell=None):
    """
    Skips execution of current line/cell if line evaluates to True.
    """
    if eval(line):
        return

    get_ipython().run_cell(cell)


def load_ipython_extension(shell):
    """
    Registers the skip magics when the extension loads.
    """
    shell.register_magic_function(skip, 'line_cell')


def unload_ipython_extension(shell):
    """
    Un-registers the skip magics when the extension unloads.
    """
    del shell.magics_manager.magics['cell']['skip']