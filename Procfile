web: daphne -b 0.0.0.0 -p $PORT config.asgi:application
worker: celery -A config worker -l info --pool=threads --concurrency=${CELERY_WORKER_CONCURRENCY:-8} -B
release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
