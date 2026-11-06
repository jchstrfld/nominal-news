FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
COPY requirements-reader.txt .
RUN pip install --no-cache-dir -r requirements-reader.txt \
    && python -m pip uninstall --yes pip \
    && python -c "import pathlib, shutil, sysconfig; shutil.rmtree(pathlib.Path(sysconfig.get_path('stdlib')) / 'ensurepip')" \
    && groupadd --gid 10001 reader \
    && useradd --uid 10001 --gid reader --no-create-home reader
COPY main.py reader_security.py edition_intro.py ./
COPY templates/ ./templates/
COPY static/style.css ./static/style.css
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--no-proxy-headers", "--limit-concurrency", "50", "--timeout-keep-alive", "5"]
