from kedro.pipeline import Pipeline, pipeline, node


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
    ])
