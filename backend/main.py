from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
import logging
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import time

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter(
    "model_requests_total",
    "Total number of model requests",
    ["status"]
)

REQUEST_LATENCY = Histogram(
    "model_request_latency_seconds",
    "Latency of model inference"
)

class InputText(BaseModel):
    text: str

# Глобальная переменная для модели
_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    
    # ИНИЦИАЛИЗАЦИЯ ПРИ СТАРТЕ
    logger.info("🚀 Starting application initialization...")
    
    try:
        # Импортируем здесь, чтобы избежать ранней загрузки
        from model import NERClassifier
        
        logger.info("📥 Loading NER model...")
        _model = NERClassifier()
        _model.load_state_dict(torch.load('model_state_dict.pt', map_location='cpu'))
        logger.info("✅ Model loaded successfully!")
        
        # Проверяем что модель работает
        test_result = _model.predict("тестовый текст")
        logger.info(f"🧪 Test inference completed, result shape: {len(test_result)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        _model = None
        raise
    
    # Приложение запускается
    yield
    
    # ОЧИСТКА ПРИ ЗАВЕРШЕНИИ
    logger.info("🛑 Shutting down application...")
    _model = None
    logger.info("✅ Cleanup completed")

# Создаем приложение с lifespan менеджером
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Обработчик ошибок валидации"""
    return JSONResponse(
        status_code=400,
        content={
            "detail": exc.errors(),
            "message": "Validation error"
        }
    )

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "NER Model API",
        "endpoints": {
            "POST /forward": "Run model on text"
        }
    }

@app.post("/forward")
async def forward(text: InputText):
    global _model
    start_time = time.time()

    try:
        result = _model.predict(text.text)

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.time() - start_time)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        REQUEST_COUNT.labels(status="failed").inc()

        return JSONResponse(
            status_code=403,
            content={"message": "модель не смогла обработать данные"}
        )

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")