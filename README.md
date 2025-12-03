MaxBank

AI-система анализа тональности банковских отзывов

О проекте

Веб-приложение для автоматического определения тональности отзывов о банках. Использует нейронную сеть для классификации текста на три категории: позитивный, нейтральный, негативный.

Ключевые особенности:
AI-модель: Нейросеть на Keras для анализа текста
Два интерфейса: Форма для клиентов + панель аналитика
Реальная аналитика: Диаграммы и статистика в реальном времени
REST API: Готовый API для интеграции

Структура проекта
```bash
maxbank/      
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   ├── app.py
│   ├── model/
│   │   ├── label_encoder.joblib
│   │   ├── model.keras
│   │   └── tfidf_vectorizer.pkl
│   └── templates/
│       ├── analytics.html
│       └── index.html
└── training/
    └── final_work.py
```
Запуск
```bash
1. Распаковать архив
2. Открыть терминал в этой папке 
3. Создать 
   python -m venv venv
4. Активировать окружение 
   venv\Scripts\activate
5. Установить зависимости
   pip install -r requirements.txt
6. Перейти в папку app и запустить
   cd app
   python app.py
7. Открыть в браузере:
   http://localhost:5000
   http://localhost:5000/analytics
