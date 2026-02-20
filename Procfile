web: daphne -b 0.0.0.0 -p $PORT config.asgi:application
release: python manage.py migrate --noinput && python manage.py collectstatic --noinput
