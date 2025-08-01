FROM condaforge/miniforge3:25.3.1-0

WORKDIR /tmp

COPY environment.yaml /tmp/environment.yaml

RUN mamba env create -f environment.yaml && \
    mamba clean -afy && \
    rm /tmp/environment.yaml

COPY src/ /pge/src/
COPY run_batch.sh /pge/

WORKDIR /pge/

ENTRYPOINT ["conda", "run", "-n", "cf2zarr", "/pge/run_batch.sh"]


