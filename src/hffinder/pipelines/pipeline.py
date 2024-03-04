from kedro.pipeline import Pipeline, pipeline, node
from .data import preprocess, prepare_for_tokenizer
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
            inputs=[
                'dataset#hf',
                'params:preprocess.url_token',
                'params:preprocess.n_jobs'
            ],
            outputs="clean_dataset#hf",
            name='preprocess',
            namespace='datasets',
        ),
        node(
            prepare_for_tokenizer,
            inputs=[
                "sentence_transformer",
                "clean_dataset#hf",
                "params:prompt.context",
                'params:preprocess.n_jobs'
            ],
            outputs="processed_dataset#hf",
            name='prepare_for_tokenizer',
            namespace='datasets',
        ),
        node(
            compute_embeddings,
            inputs=[
                "clean_dataset#hf",
                "sentence_transformer",
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
            inputs=[
                'model#hf',
                'params:preprocess.url_token',
                'params:preprocess.n_jobs'
            ],
            outputs="clean_model#hf",
            name='preprocess',
            namespace='models',
        ),
        node(
            prepare_for_tokenizer,
            inputs=[
                "sentence_transformer",
                "clean_model#hf",
                "params:prompt.context",
                'params:preprocess.n_jobs'
            ],
            outputs="processed_model#hf",
            name='prepare_for_tokenizer',
            namespace='models',
        ),
        node(
            compute_embeddings,
            inputs=[
                "clean_model#hf",
                "sentence_transformer",
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
