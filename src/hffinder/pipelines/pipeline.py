from kedro.pipeline import Pipeline, pipeline, node
from .data import preprocess_models, preprocess_datasets


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
            preprocess_models,
            inputs='model#hf',
            outputs="clean_model#hf",
            name='preprocess_models'
        ),
        node(
            preprocess_datasets,
            inputs='dataset#hf',
            outputs="clean_dataset#hf",
            name='preprocess_datasets'
        ),
    ])
