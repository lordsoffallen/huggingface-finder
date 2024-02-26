from kedro.pipeline import Pipeline, pipeline, node
from .data import preprocess
from .embeddings import compute_embeddings


def create_datasets_pipeline() -> Pipeline:
    return pipeline([
        node(
            lambda x: x,
            inputs='dataset#api',
            outputs="dataset#hf",
            name='fetch',
            namespace='datasets',
        ),
        node(
            preprocess,
            inputs=['dataset#hf', 'params:preprocess.n_jobs'],
            outputs="clean_dataset#hf",
            name='preprocess',
            namespace='datasets',
        ),
        node(
            compute_embeddings,
            inputs=[
                "clean_dataset#hf",
                "embeddings_model",
                'params:embeddings.batch_size'
            ],
            outputs='dataset_embeddings#hf',
            name='compute_embeddings',
            namespace='datasets'
        ),
    ])


def create_models_pipeline() -> Pipeline:
    return pipeline([
        node(
            lambda x: x,
            inputs='model#api',
            outputs="model#hf",
            name='fetch',
            namespace='models',
        ),
        node(
            preprocess,
            inputs=['model#hf', 'params:preprocess.n_jobs'],
            outputs="clean_model#hf",
            name='preprocess_models',
            namespace='models',
        ),
        node(
            compute_embeddings,
            inputs=[
                "clean_model#hf",
                "embeddings_model",
                'params:embeddings.batch_size'
            ],
            outputs='model_embeddings#hf',
            name='compute_embeddings',
            namespace='models',
        ),
    ])


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        create_datasets_pipeline(),
        create_models_pipeline(),
    ])
