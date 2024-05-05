import sys

from huggingface_hub import notebook_login
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.framework.project import settings
from kedro.config import OmegaConfigLoader
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path


PROJECT_PATH = Path.joinpath(Path.cwd(), 'huggingface-finder')


def setup_kedro(minimal: bool = True):
    """If minimal setup is desired, make kedro runnable via session
    otherwise return kedro context"""

    # Add cloned directory to sys path to be able to load libs/packages
    sys.path.extend([
        '/kaggle/working/huggingface-finder',
        '/kaggle/working/huggingface-finder/',
        '/kaggle/working/huggingface-finder/src',
        '/kaggle/working/huggingface-finder/src/'
        ])

    print("Setting up kedro metadata")
    # Setup kedro
    metadata = bootstrap_project(PROJECT_PATH)
    print(metadata)

    if not minimal:
        print('Creating kedro context')

        kedro_context = KedroContext(
            package_name=metadata.package_name,
            project_path=metadata.project_path,
            config_loader=OmegaConfigLoader(
                conf_source=str(metadata.project_path / settings.CONF_SOURCE)
            ),
            hook_manager=_create_hook_manager(),
            env=None,
        )
        print('Defined catalog is as follows: ')
        print(kedro_context.catalog.list())

        return kedro_context


def run_kedro(node: str, namespace: str = None, extra_params: dict = None):
    with KedroSession.create(PROJECT_PATH, extra_params=extra_params) as session:
        session.run(
            node_names=[f'{namespace}{"." if namespace is not None else ""}{node}']
        )
