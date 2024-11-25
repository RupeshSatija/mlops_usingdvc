# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . /app

# Install the project and its dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Final stage
FROM python:3.12-slim-bookworm

# Create a non-root user
# RUN useradd -m app

# Create necessary directories and set permissions
# RUN mkdir -p /app && \
#     chown -R app:app /app && \
#     chmod -R 777 /app  # Changed to 777 to allow full permissions

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app

# Set the working directory
WORKDIR /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
# USER app

# Set up the entrypoint to activate the virtual environment
ENTRYPOINT ["/bin/bash", "-c", "source .venv/bin/activate && exec $0 $@"]

# Run the application. Train the model.
CMD ["python", "src/train.py"]
