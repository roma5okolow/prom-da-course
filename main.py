from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
import logging

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        _model.load_state_dict(torch.load('model_state_dict.pt'))
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

@app.post("/forward")
async def forward(text: InputText):
    global _model
    
    try:
        logger.info(f"📨 Processing text: {text.text[:50]}...")
        result = _model.predict(text.text)
                
        return {
            "success": True,
            "result": result,
            "input_length": len(text.text)
        }
        
    except Exception as e:
        logger.error(f"❌ Модель не смогла обработать данные: {e}")
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": str(e),
                "input_text": text.text[:100]
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