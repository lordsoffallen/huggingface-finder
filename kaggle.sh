#!  /bin/sh

pip3 install -q -U xformers --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -q \
        accelerate \
        datasets==2.17.0 \
        kedro \
        kedro_datasets \
        pregex \
        python-iso639 \
        markdownify
