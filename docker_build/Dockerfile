FROM tensorflow/tensorflow:latest
LABEL maintainer="Reck Mi"

# Pillow needs libjpeg by default as of 3.0.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg8-dev \
        && apt-get install -y python3-pip && \
		python3 -m pip install ipykernel && \
		python3 -m ipykernel install --user && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install scikit-learn pyreadline Pillow
RUN rm -rf /notebooks/
VOLUME /notebooks
WORKDIR /notebooks
CMD ["/run_jupyter.sh", "--allow-root"]