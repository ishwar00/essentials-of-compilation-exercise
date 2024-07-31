FROM --platform=linux/amd64 gcc:13-bookworm
RUN apt-get update && apt-get install -y python3-pip python3-graphviz graphviz
WORKDIR /eoc
ENTRYPOINT [ "/bin/bash" ]
