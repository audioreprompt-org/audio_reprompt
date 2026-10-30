# -------------------------------------------------------
# PostgreSQL + pgvector
# -------------------------------------------------------
# Based on the official pgvector image which ships the
# 'vector' extension pre-compiled for the matching PG version.
# https://hub.docker.com/r/pgvector/pgvector
# -------------------------------------------------------
FROM pgvector/pgvector:pg17

# Forward project env vars to the standard PG env vars
# that the entrypoint uses to bootstrap the cluster.
ENV POSTGRES_USER=${DB_USER:-postgres}
ENV POSTGRES_PASSWORD=${DB_PASSWORD:-postgres}
ENV POSTGRES_DB=${DB_NAME:-postgres}

# Expose the default PostgreSQL port
EXPOSE 5432
