# Оценка риска банкротства

Демонстрационное приложение для просмотра прогнозов обученной модели по `index`, JSON-признакам или одной строке CSV.

## Требования

- Node.js 24
- Python 3.12
- pnpm
- `uv` для установки Python-зависимостей

## Запуск

```bash
pnpm install
pnpm --filter @workspace/api-server run dev
```

В отдельном терминале запустите веб-интерфейс:

```bash
pnpm --filter @workspace/risk-estimator run dev
```

Проверка типов:

```bash
pnpm run typecheck
```

API самостоятельно запускает внутренний Python-сервис инференса; второй экземпляр сервиса запускать не нужно. Python-зависимости перечислены в `pyproject.toml` и `uv.lock`.

## Модель и облачное хранилище

Если файлы модели присутствуют в `artifacts/api-server/ml/artifacts`, API использует их локально. Если локальных файлов нет, задайте `MODEL_ASSETS_GCS_URI` в формате `gs://bucket-name/optional-prefix` и настройте стандартные Google Cloud Application Default Credentials для среды запуска. Префикс модели и её версия задаются в загрузчике и API-модуле модели.

Загрузка файлов модели в Google Cloud Storage:

```bash
pnpm --filter @workspace/scripts run upload-model-assets
```

Команда использует те же `MODEL_ASSETS_GCS_URI` и Application Default Credentials.

## Структура

- `artifacts/api-server/src/routes/predictions.ts` — API прогнозов
- `artifacts/api-server/ml/predictor.py` — обработка признаков и инференс
- `artifacts/api-server/src/lib/modelAssets.ts` — получение файлов модели
- `artifacts/risk-estimator/` — русскоязычный веб-интерфейс
- `lib/api-spec/openapi.yaml` — источник схемы API

## Ограничения модели

- CSV-инференс ожидает ровно 1367 числовых значений в порядке `feature_0`–`feature_1366`; пустые значения считаются пропущенными.
- Архив содержит кэшированные прогнозы по `index`, но не исходные строки TRAIN/TEST. Для них доступны вероятности и текстовые вклады; SHAP-значения отдельных признаков недоступны без исходных строк.
- Прогнозы — только техническая демонстрация модели. Не используйте их для кредитных решений или иных реальных финансовых решений.
