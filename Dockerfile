FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "signal_bot_v2.py", "--pair", "EURUSD", "--duration", "86400", "--interval", "300"]
