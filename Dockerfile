FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8501

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade "pip==26.2.1" \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "exec streamlit run AdquitySEM.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
