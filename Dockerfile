# GPU image for the relocalization service and map building.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
# hloc must be cloned recursively (SuperPoint/SuperGlue live in git submodules).
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization.git hloc \
    && pip install --no-cache-dir -e hloc

COPY . /opt/aria-hloc
RUN pip install --no-cache-dir -e "/opt/aria-hloc[aria,service]"

EXPOSE 8080
ENTRYPOINT ["aria-hloc"]
CMD ["serve", "--map", "/maps/map", "--port", "8080"]
