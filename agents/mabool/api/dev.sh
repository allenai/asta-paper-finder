#!/bin/bash
gunicorn -k uvicorn.workers.UvicornWorker \
    --workers 1 \
    --timeout 0 \
    --bind 0.0.0.0:8000 \
    --enable-stdio-inheritance \
    --access-logfile \
    - --reload \
    --env 'APP_CONFIG_ENV=dev' \
    'mabool.api.app:create_app()'
