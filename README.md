# credit_scoring
credit scoring

Feature_selection.ipynb - отбор фичей для обучения модели

Train_model.ipynb - обучение модели, обработка результатов и создание папки model

./service -  папка с микросервисом

./test_data - папка с 25 строками тестовых данных

test_curl.sh - скрипт для проверки работоспособности сервиса

запуск микросервиса uvicorn service.main:app --port 8000 --host 0.0.0.0

запуск  docker compose up --build -d сборка и запуск контейнеров (контейнер прокинут на localhost:8000 )

Метод микросервиса по тестовому показу работы и требованиям к input_data /get_test_resolution

Метод который отдает результат /get_resolution


## 🚀 Запуск микросервиса

### 1. Локальный запуск
```bash
uvicorn service.main:app --host 0.0.0.0 --port 8000
```

После запуска сервис будет доступен по адресу:
```
http://localhost:8000
```

---

### 2. Запуск через Docker Compose
```bash
docker compose up --build -d
```

После сборки контейнер будет доступен по адресу:
```
http://localhost:8000
```

---

## 🧠 Методы микросервиса

| Метод | Описание | Пример запроса |
|-------|-----------|----------------|
| `GET /get_test_resolution` | Тестовый метод для демонстрации работы и проверки структуры `input_data`. | `curl http://localhost:8000/get_test_resolution` |
| `POST /get_resolution` | Возвращает результат скоринговой модели для переданных данных. | `curl -X POST -H "Content-Type: application/json" -d @test_data/sample.json http://localhost:8000/get_resolution` |

---

## 🧪 Проверка работы

Для быстрой проверки микросервиса можно использовать скрипт:
```bash
bash test_curl.sh
```

