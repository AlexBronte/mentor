# Проверка исправленного агента — 20 октября 2026

## Реальный end-to-end

Команда: `.venv/Scripts/python.exe main.py`.
Groq: `openai/gpt-oss-120b`; MCP SDK 2.3.0, отдельный stdio-сервер.
Вход: `data/test_articles.json`, 10 статей. Выход: `data/result.json`.
Завершение: код 0; 21:17:27 по Москве, `Complete: 10/10`.

Модель сама сформировала 32 MCP-вызова:
- open_articles — 1;
- read_article — 10;
- summarize_article — 10;
- append_result — 10;
- finish — 1.

Успех подтверждён finish после проверки файла сервером. Дополнительная проверка
выходного JSON: 10 записей, уникальные id в исходном порядке, точные id/title,
непустые строковые summary, ровно поля id/title/summary.
Полный локальный журнал: logs/live-agent-tools.log (не публикуется).

## Автоматические тесты

`python -m unittest discover -s tests -v`: 13 тестов, OK.
`python -m pip check`: No broken requirements found.

Проверены: доступность модели всех пяти инструментов, выполнение всех выбранных
действий, обратный порядок статей (граф не навязывает индекс), максимум четыре
сообщения без текстов статей в контексте планировщика, исправление раннего finish,
остановка после трёх ошибок, ограничение бесконечных повторов, пустой вход,
резервная копия, частичный результат при сбое суммаризации, валидация входа,
защита от сохранения до суммаризации, повторные вызовы без дубликатов,
реальный stdio-процесс, атомарная запись при ошибке диска, неверные пути,
отказ сохранять обрезанный ответ модели.

Тестовые модели используются только в tests/. Реальный прогон выше использует
Groq и для выбора инструментов, и для суммаризации.

## Ограничения

Нет автоматического resume и блокировки нескольких процессов с одним выходом.
Фактическая точность резюме не оценивалась экспертной разметкой или LLM judge.
Проверки подтверждают структуру результата и исполнение инструментов моделью,
но не гарантируют безошибочную работу модели при любых входных данных.

## Трасса реальных решений модели

```text
open_articles {}
read_article {"index": 0}
summarize_article {}
append_result {"index": 0}
read_article {"index": 1}
summarize_article {}
append_result {"index": 1}
read_article {"index": 2}
summarize_article {}
append_result {"index": 2}
read_article {"index": 3}
summarize_article {}
append_result {"index": 3}
read_article {"index": 4}
summarize_article {}
append_result {"index": 4}
read_article {"index": 5}
summarize_article {}
append_result {"index": 5}
read_article {"index": 6}
summarize_article {}
append_result {"index": 6}
read_article {"index": 7}
summarize_article {}
append_result {"index": 7}
read_article {"index": 8}
summarize_article {}
append_result {"index": 8}
read_article {"index": 9}
summarize_article {}
append_result {"index": 9}
finish {}
```
