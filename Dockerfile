FROM --platform=linux/amd64 python:3.12-bookworm
RUN apt-get update && apt-get install -y gcc graphviz
RUN pip install graphviz
WORKDIR /eoc
ENTRYPOINT [ "/bin/bash" ]
