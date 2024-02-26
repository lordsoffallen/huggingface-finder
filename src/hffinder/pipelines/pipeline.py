from kedro.pipeline import Pipeline, pipeline, node
from .data import preprocess


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            lambda x: x,
            inputs='dataset#api',
            outputs="dataset#hf",
            name='fetch_datasets'
        ),
        node(
            lambda x: x,
            inputs='model#api',
            outputs="model#hf",
            name='fetch_models'
        ),
        node(
            preprocess,
            inputs=['dataset#hf', 'params:preprocess.n_jobs'],
            outputs="clean_dataset#hf",
            name='preprocess_datasets'
        ),
        node(
            preprocess,
            inputs=['model#hf', 'params:preprocess.n_jobs'],
            outputs="clean_model#hf",
            name='preprocess_models'
        ),
    ])
