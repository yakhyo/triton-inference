from celery import Celery

BROKER_URI = "amqp://rabbitmq"
BACKEND_URI = "redis://redis"

app = Celery(
    "worker",
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=["worker.tasks"]  # <-- clearly
)
