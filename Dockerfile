# Stage 1: Build the Rust backend
FROM rust:1.95-slim-bookworm AS builder
WORKDIR /app
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY backend/ .
RUN cargo build --release

# Stage 2: Distroless runtime — no shell, no package manager, minimal CVE surface
FROM gcr.io/distroless/cc-debian12
WORKDIR /app
# Copy the binary
COPY --from=builder /app/target/release/retrieval-backend ./retrieval-backend
# Copy OpenSSL shared libs (reqwest links against these; distroless/cc does not include them)
COPY --from=builder /usr/lib/x86_64-linux-gnu/libssl.so.3    /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcrypto.so.3 /usr/lib/x86_64-linux-gnu/
EXPOSE 8000
CMD ["./retrieval-backend"]
