import os

from invoke import task


@task
def python(ctx):
    """ """
    ctx.run("which python" if os.name != "nt" else "where python")
