"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``pytest`` from the project root directory.
"""

from pathlib import Path

import pytest

from kedro.config import OmegaConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.project import settings


PROJECT_PATH = Path.cwd().parent


@pytest.fixture
def config_loader():
    return OmegaConfigLoader(conf_source=str(PROJECT_PATH / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="hffinder",
        project_path=PROJECT_PATH,
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
        env=None,
    )


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:
    def test_project_path(self, project_context):
        assert project_context.project_path == PROJECT_PATH

    def test_catalog_load(self, project_context):
        catalog = project_context.catalog.list()
        print(catalog)
