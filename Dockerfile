# Stage 1: Build the Rust backend
FROM rust:1.85-slim-bookworm AS builder
WORKDIR /app
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY backend/ .
RUN cargo build --release

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/retrieval-backend ./retrieval-backend
EXPOSE 8000
CMD ["./retrieval-backend"]
