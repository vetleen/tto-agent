web: bin/start-pgbouncer daphne -b 0.0.0.0 -p $PORT config.asgi:application
worker: RERANK_ENABLED=${RERANK_ON_WORKER:-false} bin/start-pgbouncer celery -A config worker -l info --pool=threads --concurrency=${CELERY_WORKER_CONCURRENCY:-8} -B
release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
