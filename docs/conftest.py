"""Run the Python code blocks of the tutorials in docs/source as tests.

The code blocks of every .rst file are executed in order in a shared namespace, so later
blocks can use objects defined in earlier ones. A block that is not meant to be run, for
example because it contains placeholders, can be excluded with the reST comment
``.. skip: next`` directly in front of it.
"""

import os
import tempfile

import matplotlib
import torch
from sybil import Sybil
from sybil.parsers.rest import PythonCodeBlockParser, SkipParser

# plots are only created, never shown
matplotlib.use("Agg")


def setup(namespace):
    # files written by the tutorials end up in a temporary directory
    namespace["_tmpdir"] = tempfile.TemporaryDirectory()
    namespace["_old_cwd"] = os.getcwd()
    os.chdir(namespace["_tmpdir"].name)
    torch.manual_seed(1234)


def teardown(namespace):
    import matplotlib.pyplot as plt

    plt.close("all")
    os.chdir(namespace["_old_cwd"])
    namespace["_tmpdir"].cleanup()


pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser(), SkipParser()],
    patterns=["*.rst"],
    setup=setup,
    teardown=teardown,
).pytest()
