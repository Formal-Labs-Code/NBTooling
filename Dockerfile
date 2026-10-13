# syntax=docker/dockerfile:1

# ---- build stage: has Python + Node, builds nbcore + NBFix + the JupyterLab labextension into a venv ----
FROM python:3.11-slim AS build

# Node.js is only needed here to build jupyterlab-nbfix's TypeScript
# source into static assets - it never ends up in the runtime image below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /src
# nbcore and nbfix are separate installable packages (see each
# directory's own pyproject.toml) - nbfix declares nbcore as a
# dependency, but since nbcore isn't published anywhere, both local
# paths must be passed to the same pip install so the resolver can
# satisfy that dependency from the local copy instead of failing to
# find "nbcore" on an index. jupyter-config/ now lives inside
# nbfix/ and installs automatically via its own pyproject.toml
# shared-data config - no separate COPY needed for it.
COPY nbcore ./nbcore
COPY nbfix ./nbfix
COPY jupyterlab-nbfix ./jupyterlab-nbfix

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ./nbcore "./nbfix[jupyter]" "jupyterlab>=4,<5" \
    && pip install --no-cache-dir ./jupyterlab-nbfix

# ---- runtime stage: slim, no Node/npm/node_modules/build toolchain ----
FROM python:3.11-slim AS runtime

RUN useradd --create-home --shell /bin/bash nbfix
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

USER nbfix
WORKDIR /home/nbfix/work
EXPOSE 8888

# No token/password is baked in here on purpose - JupyterLab generates a
# fresh one at startup and prints the URL (with token) to the container
# logs. Run `docker logs <container>` to get the link to open.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
