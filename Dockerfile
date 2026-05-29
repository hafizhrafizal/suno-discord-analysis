# Stage 1: Build the Rust backend
FROM rust:1.78-slim-bookworm AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY backend/ .
RUN cargo build --release

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/retrieval-backend ./retrieval-backend
EXPOSE 8000
CMD ["./retrieval-backend"]
