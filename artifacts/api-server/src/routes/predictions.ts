import { Router, type IRouter } from "express";
import {
  GetModelInfoResponse,
  PredictByIndexBody,
  PredictByIndexResponse,
  PredictFromCsvResponse,
  PredictFromFeaturesBody,
  PredictFromFeaturesResponse,
} from "@workspace/api-zod";
import { MlServiceError, mlService } from "../lib/mlService";

const router: IRouter = Router();
const FEATURE_COUNT = 1367;

function isValidFeatureName(name: string): boolean {
  const match = /^feature_(\d+)$/.exec(name);
  if (!match?.[1]) return false;
  const index = Number(match[1]);
  return Number.isInteger(index) && index >= 0 && index < FEATURE_COUNT;
}

function parseCsvFeatures(input: unknown): Record<string, number | null> {
  if (typeof input !== "string") {
    throw new Error("Тело запроса должно содержать одну строку text/csv.");
  }

  const row = input.replace(/^\uFEFF/, "").trim();
  if (!row || /[\r\n]/.test(row)) {
    throw new Error("CSV должен содержать ровно одну строку без заголовка.");
  }

  const values = row.split(",");
  if (values.length !== FEATURE_COUNT) {
    throw new Error(
      `В CSV должно быть ровно ${FEATURE_COUNT} значений в порядке feature_0–feature_1366.`,
    );
  }

  const features: Record<string, number | null> = {};
  for (let index = 0; index < values.length; index += 1) {
    const rawValue = values[index]?.trim() ?? "";
    if (!rawValue) {
      features[`feature_${index}`] = null;
      continue;
    }

    const value = Number(rawValue);
    if (!Number.isFinite(value)) {
      throw new Error(`Значение feature_${index} должно быть числом или пустым.`);
    }
    features[`feature_${index}`] = value;
  }

  return features;
}

function modelUnavailableMessage(error: string | null): string {
  if (error) return `Модель недоступна: ${error}`;
  return "Модель загружается. Повторите запрос через несколько секунд.";
}

router.get("/model-info", (_req, res): void => {
  const status = mlService.status();
  const info = status.info;
  const ready = status.ready;
  const indexReady = ready && Boolean(info?.index_lookup);

  let message: string;
  if (!ready) {
    message = modelUnavailableMessage(status.error);
  } else if (!indexReady) {
    message =
      "Модель загружена, но готовые прогнозы по index недоступны.";
  } else if (!info?.index_explanations) {
    message =
      "Модель загружена. Прогноз по index доступен; исходные признаки для SHAP-объяснения не входят в архив.";
  } else {
    message = "Модель и прогнозы по index готовы.";
  }

  const response = GetModelInfoResponse.parse({
    ready,
    model: info?.model ?? "unavailable",
    model_version: info?.version ?? "not-loaded",
    oof_auc: info?.oof_auc ?? null,
    feature_count: info?.n_features ?? 0,
    index_lookup_ready: indexReady,
    dataset_ready: indexReady,
    message,
  });
  res.json(response);
});

router.post("/predictions/by-index", async (req, res): Promise<void> => {
  const parsed = PredictByIndexBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({
      error: 'Укажите dataset ("train" или "test") и неотрицательный целый index.',
    });
    return;
  }

  try {
    const raw = await mlService.predictByIndex(parsed.data.index);
    const result = PredictByIndexResponse.parse(raw);
    if (result.source !== parsed.data.dataset) {
      res.status(404).json({
        error: `Индекс ${parsed.data.index} не найден в наборе ${parsed.data.dataset}.`,
      });
      return;
    }
    res.json(result);
  } catch (error) {
    if (error instanceof MlServiceError) {
      res.status(error.status).json({ error: error.message });
      return;
    }
    req.log.error({ err: error }, "Prediction by index failed");
    res.status(500).json({ error: "Не удалось получить прогноз по index." });
  }
});

router.post("/predictions", async (req, res): Promise<void> => {
  const parsed = PredictFromFeaturesBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Проверьте формат признаков в запросе." });
    return;
  }

  const names = Object.keys(parsed.data.features);
  if (names.length === 0 || names.length > FEATURE_COUNT) {
    res.status(400).json({
      error: `Передайте от 1 до ${FEATURE_COUNT} табличных признаков.`,
    });
    return;
  }

  const invalidName = names.find((name) => !isValidFeatureName(name));
  if (invalidName) {
    res.status(400).json({
      error: `Неизвестное имя признака: ${invalidName}. Ожидается feature_0–feature_1366.`,
    });
    return;
  }

  try {
    const raw = await mlService.predictFromFeatures(parsed.data);
    res.json(PredictFromFeaturesResponse.parse(raw));
  } catch (error) {
    if (error instanceof MlServiceError) {
      res.status(error.status).json({ error: error.message });
      return;
    }
    req.log.error({ err: error }, "Feature prediction failed");
    res.status(500).json({ error: "Не удалось получить прогноз по признакам." });
  }
});

router.post("/predictions/from-csv", async (req, res): Promise<void> => {
  let features: Record<string, number | null>;
  try {
    features = parseCsvFeatures(req.body);
  } catch (error) {
    res.status(400).json({
      error: error instanceof Error ? error.message : "Некорректная CSV-строка.",
    });
    return;
  }

  try {
    const raw = await mlService.predictFromCsv(features);
    res.json(PredictFromCsvResponse.parse(raw));
  } catch (error) {
    if (error instanceof MlServiceError) {
      res.status(error.status).json({ error: error.message });
      return;
    }
    req.log.error({ err: error }, "CSV prediction failed");
    res.status(500).json({ error: "Не удалось получить прогноз из CSV." });
  }
});

export default router;
