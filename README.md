# huggingface-finder

## Overview

This is a Kedro project, which was generated using `kedro 0.19.2`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

The file in `conf/base/catalog.yml` contains configuration that can be adjusted. 

One can execute `kedro run` to run everything end to end, or via `kedro run --from-nodes=...` to run certain functions.
The workflow itself is defined in the `src/hffinder/pipelines/pipeline.py`


## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

### Conda

You can install `environment.yml` file to replicate the same environment. Additionally, 
an `environment_full.yml` is also provided which pins all the versions in the environment.

```shell
conda env create -f environment_full.yml
```

### Pip

You can install dependencies in the `environment.yml` file by passing them into pip.

## How to run a Kedro pipeline

You can run the project with: ( runs everything end to end )

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope:
> `context`, 'session', `catalog`, and `pipelines`.

### Jupyter
```
kedro jupyter notebook
```

### JupyterLab
```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout).
For example, you can add a hook in `.git/config` with `nbstripout --install`. 
This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
