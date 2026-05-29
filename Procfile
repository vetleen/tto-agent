web: daphne -b 0.0.0.0 -p $PORT config.asgi:application
worker: python -m config.celery_gevent worker -A config -l info --pool=gevent --concurrency=${CELERY_WORKER_CONCURRENCY:-8} -B
release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
