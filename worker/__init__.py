from celery import Celery

# BROKER_URI = "amqp://guest:guest@localhost:5672/"
BROKER_URI = "amqp://rabbitmq"
BACKEND_URI = "redis://redis"

app = Celery(
    "server",
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=["server.tasks"]
)
