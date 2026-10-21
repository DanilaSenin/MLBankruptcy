import { useState, type FormEvent, type ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  Activity, AlertCircle, ArrowDownRight, ArrowRight, ArrowUpRight, Check,
  ChevronDown, CircleHelp, Clipboard, Database, FileJson2, FileSpreadsheet,
  Info, LoaderCircle, RotateCw, ShieldAlert, SlidersHorizontal, Sparkles,
  Upload, X,
} from 'lucide-react';
import { useGetModelInfo, useHealthCheck, usePredictByIndex, usePredictFromCsv, usePredictFromFeatures } from '@workspace/api-client-react';
import type { ModelInfo, PredictionResult, TopFeature, TopWord } from '@workspace/api-client-react';
import { ErrorBoundary } from '@/components/error-boundary';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Route, Switch, useLocation, Router as WouterRouter } from 'wouter';

const queryClient = new QueryClient();

type Mode = 'index' | 'features' | 'csv';

const numberFormat = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 3 });
const percentFormat = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 });

function formatValue(value: number | null | undefined) {
  return value == null ? '—' : numberFormat.format(value);
}

function errorMessage(error: unknown) {
  if (error && typeof error === 'object') {
    const candidate = error as { error?: string; message?: string };
    return candidate.error || candidate.message || 'Не удалось получить оценку. Проверьте соединение и повторите попытку.';
  }
  return 'Не удалось получить оценку. Проверьте соединение и повторите попытку.';
}

function Header({ healthy }: { healthy: boolean | undefined }) {
  return (
    <header className="topbar sticky top-0 z-10 flex items-center justify-between px-5 md:px-10">
      <div className="flex items-center gap-3">
        <div className="brand-mark" aria-hidden="true" />
        <div>
          <div className="text-[13px] font-extrabold tracking-[-.04em]">контур риска</div>
          <div className="eyebrow mt-0.5 !text-[8px]">модельный стенд · 01</div>
        </div>
      </div>
      <div className="flex items-center gap-2.5 rounded-full border border-border px-3 py-2" data-testid="status-api">
        <span className={`status-dot ${healthy === false ? 'offline' : ''}`} />
        <span className="mono text-[10px] text-muted-foreground">
          {healthy === undefined ? 'ПРОВЕРКА API' : healthy ? 'API ДОСТУПЕН' : 'API НЕДОСТУПЕН'}
        </span>
      </div>
    </header>
  );
}

function Readiness({ model, isLoading, isError, onRetry }: {
  model: ModelInfo | undefined;
  isLoading: boolean;
  isError: boolean;
  onRetry: () => void;
}) {
  const rows = [
    { label: 'Модель', ok: model?.ready, note: model?.model || 'Проверяем доступность модели' },
    { label: 'Набор данных', ok: model?.dataset_ready, note: 'Для прогноза по индексу' },
    { label: 'Поиск по индексу', ok: model?.index_lookup_ready, note: 'Обучающая и тестовая выборки' },
  ];
  return (
    <section className="panel p-5 md:p-6" aria-labelledby="readiness-heading">
      <div className="mb-5 flex items-start justify-between gap-3">
        <div>
          <p className="eyebrow">Состояние системы</p>
          <h2 id="readiness-heading" className="mt-1 text-[17px] font-bold tracking-[-.04em]">Готовность модели</h2>
        </div>
        {isLoading ? <span className="mono text-[10px] text-muted-foreground">ОБНОВЛЕНИЕ…</span> :
          isError ? <button type="button" className="secondary-button !min-h-8 !px-2.5" onClick={onRetry} data-testid="button-retry-readiness"><RotateCw size={13} /> Повторить</button> :
          model?.ready ? <span className="rounded-full bg-primary/10 px-2.5 py-1 text-[10px] font-bold text-primary">СТАТУС ИЗ API</span> :
          <span className="rounded-full bg-destructive/10 px-2.5 py-1 text-[10px] font-bold text-destructive">НЕ ГОТОВА</span>}
      </div>
      {isLoading ? (
        <div className="space-y-4" aria-label="Загрузка статуса"><div className="skeleton h-8 rounded-sm" /><div className="skeleton h-8 rounded-sm" /><div className="skeleton h-8 rounded-sm" /></div>
      ) : isError ? (
        <div className="rounded-sm border border-destructive/25 bg-destructive/5 p-3 text-xs text-destructive" role="alert" data-testid="error-model-info">
          Не удалось загрузить сведения о модели. Доступность прогнозирования не подтверждена.
        </div>
      ) : (
        <>
          <div className="divide-y divide-border">
            {rows.map(({ label, ok, note }) => (
              <div className="flex items-center justify-between gap-4 py-3 first:pt-0 last:pb-0" key={label} data-testid={`readiness-${label}`}>
                <div className="min-w-0">
                  <div className="text-[12px] font-semibold">{label}</div>
                  <div className="mt-0.5 truncate text-[10px] text-muted-foreground">{note}</div>
                </div>
                {ok === undefined ? <span className="mono text-[10px] text-muted-foreground">НЕТ ДАННЫХ</span> :
                  <span className={`flex shrink-0 items-center gap-1.5 text-[10px] font-bold ${ok ? 'text-primary' : 'text-muted-foreground'}`}>
                    {ok ? <Check size={13} /> : <X size={13} />}
                    {ok ? 'ДОСТУПНО' : 'НЕДОСТУПНО'}
                  </span>}
              </div>
            ))}
          </div>
          {model?.message && <p className="mt-4 border-t border-border pt-3 text-[11px] leading-relaxed text-muted-foreground" data-testid="text-model-message">{model.message}</p>}
          <div className="mt-4 grid grid-cols-2 gap-2 border-t border-border pt-4">
            <div><div className="eyebrow !text-[8px]">Версия модели</div><div className="mono mt-1 text-[11px]">{model?.model_version || '—'}</div></div>
            <div><div className="eyebrow !text-[8px]">Признаки</div><div className="mono mt-1 text-[11px]">{model?.feature_count ?? '—'}</div></div>
            <div><div className="eyebrow !text-[8px]">OOF AUC</div><div className="mono mt-1 text-[11px]">{model?.oof_auc == null ? '—' : numberFormat.format(model.oof_auc)}</div></div>
            <div><div className="eyebrow !text-[8px]">Идентификатор</div><div className="mono mt-1 truncate text-[11px]">{model?.model || '—'}</div></div>
          </div>
        </>
      )}
    </section>
  );
}

function PredictionForm({ mode, setMode, model, onResult, onPending }: {
  mode: Mode;
  setMode: (mode: Mode) => void;
  model: ModelInfo | undefined;
  onResult: (result: PredictionResult) => void;
  onPending: (pending: boolean) => void;
}) {
  const [dataset, setDataset] = useState<'train' | 'test'>('train');
  const [indexText, setIndexText] = useState('');
  const [featureText, setFeatureText] = useState('');
  const [csvText, setCsvText] = useState('');
  const [localError, setLocalError] = useState('');
  const byIndex = usePredictByIndex();
  const fromFeatures = usePredictFromFeatures();
  const fromCsv = usePredictFromCsv();
  const activePending = mode === 'index' ? byIndex.isPending : mode === 'features' ? fromFeatures.isPending : fromCsv.isPending;
  const pending = byIndex.isPending || fromFeatures.isPending || fromCsv.isPending;
  const unavailable = mode === 'index' ? !model?.ready || !model?.dataset_ready || !model?.index_lookup_ready : !model?.ready;
  const indexValid = indexText.trim() !== '' && Number.isInteger(Number(indexText)) && Number(indexText) >= 0;

  const submitIndex = (event: FormEvent) => {
    event.preventDefault();
    setLocalError('');
    if (!indexValid) { setLocalError('Введите целый неотрицательный индекс записи.'); return; }
    onPending(true);
    byIndex.mutate({ data: { dataset, index: Number(indexText) } }, { onSuccess: data => { onResult(data); onPending(false); }, onError: () => onPending(false) });
  };
  const submitFeatures = (event: FormEvent) => {
    event.preventDefault();
    setLocalError('');
    let parsed: unknown;
    try { parsed = JSON.parse(featureText); }
    catch { setLocalError('Проверьте синтаксис JSON.'); return; }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      setLocalError('Ожидается объект с полем features.'); return;
    }
    const body = parsed as { id?: unknown; features?: unknown; text_features?: unknown };
    if (!body.features || typeof body.features !== 'object' || Array.isArray(body.features)) {
      setLocalError('Добавьте объект features с числовыми значениями признаков.'); return;
    }
    const invalid = Object.entries(body.features as Record<string, unknown>).some(([key, value]) =>
      !/^feature_\d+$/.test(key) || !(value === null || (typeof value === 'number' && Number.isFinite(value))));
    if (invalid) { setLocalError('Ключи должны иметь вид feature_N, а значения — быть числами или null.'); return; }
    if (body.text_features !== undefined && (!body.text_features || typeof body.text_features !== 'object' || Array.isArray(body.text_features) ||
      Object.values(body.text_features as object).some(value => typeof value !== 'number' || !Number.isFinite(value)))) {
      setLocalError('text_features должен быть объектом «слово: число».'); return;
    }
    if (body.id !== undefined && body.id !== null && (typeof body.id !== 'number' || !Number.isInteger(body.id))) {
      setLocalError('id должен быть целым числом или null.'); return;
    }
    onPending(true);
    fromFeatures.mutate({ data: parsed as { features: Record<string, number | null>; id?: number | null; text_features?: Record<string, number> } }, { onSuccess: data => { onResult(data); onPending(false); }, onError: () => onPending(false) });
  };
  const submitCsv = (event: FormEvent) => {
    event.preventDefault();
    setLocalError('');
    const row = csvText.trim();
    if (!row) { setLocalError('Вставьте одну строку CSV.'); return; }
    if (/[\r\n]/.test(row)) { setLocalError('Нужна ровно одна строка CSV без заголовка.'); return; }
    const cells = row.split(',');
    if (cells.length !== 1367) { setLocalError(`В строке ${cells.length} значений. Для этого формата ожидается 1 367 признаков (feature_0—feature_1366).`); return; }
    if (cells.some(value => value.trim() === '' || !Number.isFinite(Number(value)))) {
      setLocalError('Каждое значение должно быть числом; пустые поля не допускаются.'); return;
    }
    onPending(true);
    fromCsv.mutate({ data: row }, { onSuccess: data => { onResult(data); onPending(false); }, onError: () => onPending(false) });
  };

  const error = localError || (mode === 'index' && byIndex.isError ? errorMessage(byIndex.error) :
    mode === 'features' && fromFeatures.isError ? errorMessage(fromFeatures.error) :
      mode === 'csv' && fromCsv.isError ? errorMessage(fromCsv.error) : '');

  return (
    <section className="panel overflow-hidden" aria-labelledby="predict-heading">
      <div className="border-b border-border px-5 pt-5 md:px-7 md:pt-7">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="eyebrow">Исследовательский интерфейс</p>
            <h2 id="predict-heading" className="serif mt-2 text-[30px] leading-tight tracking-[-.035em] md:text-[34px]">Проверить запись</h2>
            <p className="mt-2 max-w-xl text-[12px] leading-relaxed text-muted-foreground">Получите оценку модели и разберите факторы, которые повлияли на неё. Это техническая демонстрация, а не решение о кредитоспособности.</p>
          </div>
          <div className="hidden h-11 w-11 shrink-0 items-center justify-center border border-border text-primary md:flex"><SlidersHorizontal size={18} strokeWidth={1.5} /></div>
        </div>
        <div className="mt-5 flex gap-1 overflow-x-auto" role="tablist" aria-label="Способ ввода">
          <button className="tab-button flex items-center gap-2" role="tab" aria-selected={mode === 'index'} onClick={() => { setMode('index'); setLocalError(''); }} data-testid="tab-index"><Database size={14} /> По индексу</button>
          <button className="tab-button flex items-center gap-2" role="tab" aria-selected={mode === 'features'} onClick={() => { setMode('features'); setLocalError(''); }} data-testid="tab-features"><FileJson2 size={14} /> JSON-признаки</button>
          <button className="tab-button flex items-center gap-2" role="tab" aria-selected={mode === 'csv'} onClick={() => { setMode('csv'); setLocalError(''); }} data-testid="tab-csv"><FileSpreadsheet size={14} /> Одна строка CSV</button>
        </div>
      </div>

      <div className="p-5 md:p-7">
        {mode === 'index' && (
          <form onSubmit={submitIndex} className="fade-up" data-testid="form-index">
            <div className="grid gap-5 sm:grid-cols-[1fr_1.4fr]">
              <label className="block">
                <span className="mb-2 block text-[11px] font-bold">Выборка</span>
                <span className="relative block">
                  <select className="input-field appearance-none pr-9" value={dataset} onChange={e => setDataset(e.target.value as 'train' | 'test')} data-testid="select-dataset">
                    <option value="train">Обучающая · train</option>
                    <option value="test">Тестовая · test</option>
                  </select><ChevronDown size={15} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground" />
                </span>
              </label>
              <label className="block">
                <span className="mb-2 block text-[11px] font-bold">Индекс записи</span>
                <input className="input-field mono" type="number" min="0" step="1" inputMode="numeric" placeholder="Например, индекс из вашего набора" value={indexText} onChange={e => setIndexText(e.target.value)} data-testid="input-index" />
              </label>
            </div>
            <div className="mt-4 flex items-start gap-2 text-[10px] leading-relaxed text-muted-foreground"><Info size={13} className="mt-0.5 shrink-0" /> Индексация доступна только при готовности набора данных и поиска по индексу.</div>
            <div className="mt-5 flex flex-wrap items-center gap-3">
              <button className="action-button" type="submit" disabled={pending || unavailable || !indexValid} data-testid="button-predict-index">
                {activePending ? <LoaderCircle size={15} className="animate-spin" /> : <Activity size={15} />}
                {activePending ? 'Рассчитываем…' : 'Рассчитать оценку'}
                {!activePending && <ArrowRight size={14} />}
              </button>
              {unavailable && <span className="text-[10px] text-muted-foreground">Этот способ сейчас недоступен по статусу API.</span>}
            </div>
          </form>
        )}
        {mode === 'features' && (
          <form onSubmit={submitFeatures} className="fade-up" data-testid="form-features">
            <label className="block">
              <span className="mb-2 flex items-center justify-between gap-2 text-[11px] font-bold"><span>Объект признаков</span><span className="mono text-[9px] font-normal text-muted-foreground">JSON · feature_N: число или null</span></span>
              <textarea className="input-field mono min-h-[190px] resize-y text-[11px] leading-relaxed" spellCheck={false} placeholder={'{\n  "features": {\n    "feature_0": …\n  },\n  "text_features": {\n    "слово": …\n  }\n}'} value={featureText} onChange={e => setFeatureText(e.target.value)} data-testid="input-features-json" />
            </label>
            <p className="mt-3 text-[10px] leading-relaxed text-muted-foreground">Допускаются поля <span className="mono">features</span>, необязательные <span className="mono">id</span> и <span className="mono">text_features</span>. Используйте собственные значения; демонстрация не подставляет примерные данные.</p>
            <div className="mt-5 flex flex-wrap items-center gap-3">
              <button className="action-button" type="submit" disabled={pending || unavailable || !featureText.trim()} data-testid="button-predict-features">
                {activePending ? <LoaderCircle size={15} className="animate-spin" /> : <Sparkles size={15} />}
                {activePending ? 'Рассчитываем…' : 'Рассчитать оценку'}
                {!activePending && <ArrowRight size={14} />}
              </button>
              {unavailable && <span className="text-[10px] text-muted-foreground">Модель не подтвердила готовность.</span>}
            </div>
          </form>
        )}
        {mode === 'csv' && (
          <form onSubmit={submitCsv} className="fade-up" data-testid="form-csv">
            <label className="block">
              <span className="mb-2 flex items-center justify-between gap-2 text-[11px] font-bold"><span>Строка признаков</span><span className="mono text-[9px] font-normal text-muted-foreground">без заголовка · 1 367 значений</span></span>
              <textarea className="input-field mono min-h-[150px] resize-y text-[11px] leading-relaxed" spellCheck={false} placeholder="Вставьте одну CSV-строку из вашего источника данных" value={csvText} onChange={e => setCsvText(e.target.value)} data-testid="input-csv-row" />
            </label>
            <div className="mt-3 flex items-start gap-2 text-[10px] leading-relaxed text-muted-foreground"><Upload size={13} className="mt-0.5 shrink-0" /> Одна строка без заголовка: значения для feature_0—feature_1366, разделённые запятыми.</div>
            <div className="mt-5 flex flex-wrap items-center gap-3">
              <button className="action-button" type="submit" disabled={pending || unavailable || !csvText.trim()} data-testid="button-predict-csv">
                {activePending ? <LoaderCircle size={15} className="animate-spin" /> : <FileSpreadsheet size={15} />}
                {activePending ? 'Рассчитываем…' : 'Рассчитать оценку'}
                {!activePending && <ArrowRight size={14} />}
              </button>
              {unavailable && <span className="text-[10px] text-muted-foreground">Модель не подтвердила готовность.</span>}
            </div>
          </form>
        )}
        {error && <div className="mt-5 flex items-start gap-2.5 rounded-sm border border-destructive/25 bg-destructive/5 p-3.5 text-[11px] leading-relaxed text-destructive" role="alert" data-testid="error-prediction"><AlertCircle size={15} className="mt-0.5 shrink-0" /><span>{error}</span></div>}
      </div>
    </section>
  );
}

function ProbabilityCard({ result }: { result: PredictionResult }) {
  const probability = Math.min(100, Math.max(0, result.probability * 100));
  const sourceLabel = result.source === 'train' ? 'обучающая выборка' : result.source === 'test' ? 'тестовая выборка' : 'новые признаки';
  return (
    <section className="panel overflow-hidden" aria-labelledby="probability-heading" data-testid="result-summary">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-border px-5 py-5 md:px-7 md:py-6">
        <div>
          <p className="eyebrow">Результат модели</p>
          <h2 id="probability-heading" className="serif mt-1 text-[24px] tracking-[-.03em]">Оценка вероятности</h2>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-border px-3 py-1.5 text-[9px] font-bold uppercase tracking-[.08em] text-muted-foreground">
          <span className="status-dot" /> {sourceLabel}
        </div>
      </div>
      <div className="grid gap-6 p-5 md:grid-cols-[1fr_220px] md:items-center md:px-7 md:py-7">
        <div>
          <div className="flex items-end gap-2">
            <span className="mono text-[52px] leading-none tracking-[-.08em] md:text-[62px]" data-testid="text-probability">{percentFormat.format(probability)}<span className="text-[28px] text-primary">%</span></span>
            <span className="mb-1 text-[10px] text-muted-foreground">оценка модели</span>
          </div>
          <div className="result-bar mt-5"><span style={{ width: `${probability}%` }} /></div>
          <p className="mt-3 max-w-lg text-[11px] leading-relaxed text-muted-foreground">Число отражает выход модели для переданных данных. Оно не является заключением о компании или основанием для финансового решения.</p>
        </div>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-1 md:border-l md:border-border md:pl-6">
          <div className="rounded-sm bg-muted/70 p-3">
            <div className="eyebrow !text-[8px]">Перцентиль</div>
            <div className="mono mt-1 text-[20px]" data-testid="text-percentile">{result.percentile == null ? '—' : `${percentFormat.format(result.percentile)}%`}</div>
            <div className="mt-1 text-[9px] text-muted-foreground">по шкале модели</div>
          </div>
          <div className="rounded-sm bg-muted/70 p-3">
            <div className="eyebrow !text-[8px]">ID результата</div>
            <div className="mono mt-1 truncate text-[13px]" data-testid="text-result-id">{result.id == null ? '—' : result.id}</div>
            <div className="mt-1 text-[9px] text-muted-foreground">из ответа API</div>
          </div>
        </div>
      </div>
      {result.true_target !== undefined && <div className="flex items-center gap-2 border-t border-border px-5 py-3 text-[10px] text-muted-foreground md:px-7" data-testid="text-true-target"><Info size={13} /> Известная метка в данных: <span className="mono font-medium text-foreground">{result.true_target}</span></div>}
    </section>
  );
}

function FeatureContributions({ features, base }: { features: TopFeature[]; base?: number }) {
  const max = Math.max(...features.map(item => Math.abs(item.shap_value)), 0);
  return (
    <section className="panel p-5 md:p-6" aria-labelledby="features-heading" data-testid="section-feature-impacts">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="eyebrow">Объяснение модели</p>
          <h3 id="features-heading" className="mt-1 text-[16px] font-bold tracking-[-.035em]">Вклад признаков</h3>
        </div>
        {base !== undefined && <div className="text-right"><div className="eyebrow !text-[8px]">Базовое значение</div><div className="mono mt-1 text-[11px]">{formatValue(base)}</div></div>}
      </div>
      {!features.length ? <div className="mt-5 rounded-sm border border-dashed border-border p-5 text-center text-[11px] text-muted-foreground" data-testid="empty-feature-impacts">API не вернул вкладов признаков для этого результата.</div> :
        <div className="mt-5 space-y-4">
          {features.map((item, index) => {
            const positive = item.shap_value >= 0;
            const width = max === 0 ? 0 : Math.max(2, Math.abs(item.shap_value) / max * 100);
            return (
              <div key={`${item.name}-${index}`} className="grid grid-cols-[minmax(100px,1fr)_minmax(72px,1.2fr)_66px] items-center gap-3" data-testid={`feature-impact-${index}`}>
                <div className="min-w-0">
                  <div className="truncate text-[11px] font-semibold" title={item.label}>{item.label || item.name}</div>
                  <div className="mono mt-0.5 truncate text-[9px] text-muted-foreground">{item.name} · {formatValue(item.raw_value)}</div>
                </div>
                <div className="feature-track"><span className={positive ? 'positive' : 'negative'} style={{ width: `${width}%` }} /></div>
                <div className={`mono flex items-center justify-end gap-1 text-[10px] ${positive ? 'text-destructive' : 'text-primary'}`}>
                  {positive ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                  {formatValue(item.shap_value)}
                </div>
              </div>
            );
          })}
          <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-border pt-3 text-[9px] text-muted-foreground">
            <span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-sm bg-destructive/80" /> повышает выход модели</span>
            <span className="flex items-center gap-1.5"><i className="h-2 w-2 rounded-sm bg-primary/80" /> снижает выход модели</span>
          </div>
          <p className="text-[9px] leading-relaxed text-muted-foreground">Знак и величина вклада показаны в том виде, в котором их вернул API; это объяснение модели, а не причинный вывод.</p>
        </div>}
    </section>
  );
}

function WordImpacts({ words }: { words: TopWord[] }) {
  return (
    <section className="panel p-5 md:p-6" aria-labelledby="words-heading" data-testid="section-word-impacts">
      <div className="flex items-start justify-between gap-3">
        <div><p className="eyebrow">Текстовые признаки</p><h3 id="words-heading" className="mt-1 text-[16px] font-bold tracking-[-.035em]">Влияние слов</h3></div>
        <span className="rounded-full bg-muted px-2 py-1 mono text-[9px] text-muted-foreground">{words.length} {words.length === 1 ? 'слово' : 'слов'}</span>
      </div>
      {!words.length ? <div className="mt-5 rounded-sm border border-dashed border-border p-5 text-center text-[11px] text-muted-foreground" data-testid="empty-word-impacts">API не вернул текстовых вкладов для этого результата.</div> :
        <div className="mt-4 divide-y divide-border">
          {words.map((word, index) => {
            const positive = word.impact >= 0;
            return (
              <div key={`${word.word}-${index}`} className="grid grid-cols-[1fr_auto] gap-2 py-3 first:pt-1 last:pb-0" data-testid={`word-impact-${index}`}>
                <div className="min-w-0">
                  <div className="flex items-baseline gap-2"><span className="truncate text-[12px] font-semibold">{word.word}</span><span className="mono shrink-0 text-[9px] text-muted-foreground">×{formatValue(word.count)}</span></div>
                  <div className="mt-1 text-[9px] text-muted-foreground">банкротство {word.share_bankrupt == null ? '—' : `${percentFormat.format(word.share_bankrupt)}%`} <span className="mx-1 opacity-50">/</span> без метки {word.share_nonbankrupt == null ? '—' : `${percentFormat.format(word.share_nonbankrupt)}%`}</div>
                </div>
                <div className={`mono self-center text-[10px] ${positive ? 'text-destructive' : 'text-primary'}`}>{positive ? '+' : ''}{formatValue(word.impact)}</div>
              </div>
            );
          })}
        </div>}
    </section>
  );
}

function Warnings({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null;
  return (
    <section className="border border-amber-700/20 bg-amber-500/5 p-4 md:p-5" aria-label="Предупреждения API" data-testid="section-warnings">
      <div className="flex items-center gap-2 text-[11px] font-bold"><ShieldAlert size={14} className="text-amber-800" /> Замечания к результату</div>
      <ul className="mt-2 space-y-1.5 pl-5 text-[10px] leading-relaxed text-muted-foreground">{warnings.map((warning, i) => <li key={`${warning}-${i}`}>{warning}</li>)}</ul>
    </section>
  );
}

function ResultPanel({ result }: { result: PredictionResult }) {
  return (
    <div className="fade-up space-y-4" aria-live="polite">
      <ProbabilityCard result={result} />
      <div className="grid gap-4 lg:grid-cols-[1.15fr_.85fr]">
        <FeatureContributions features={result.top_features || []} base={result.shap_base} />
        <WordImpacts words={result.top_words || []} />
      </div>
      <Warnings warnings={result.warnings || []} />
      <div className="flex items-start gap-2 border-l-2 border-primary/40 py-1 pl-3 text-[10px] leading-relaxed text-muted-foreground">
        <CircleHelp size={13} className="mt-0.5 shrink-0" />
        <p>Результат предназначен для изучения поведения модели. Он не заменяет проверку данных, профессиональный анализ или решение уполномоченного специалиста.</p>
      </div>
    </div>
  );
}

function Home() {
  const [mode, setMode] = useState<Mode>('index');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [inProgress, setInProgress] = useState(false);
  const modelQuery = useGetModelInfo<ModelInfo>();
  const healthQuery = useHealthCheck();
  const response = result;

  return (
    <div className="app-shell grain">
      <Header healthy={healthQuery.data ? healthQuery.data.status === 'ok' : healthQuery.isError ? false : undefined} />
      <main className="mx-auto w-full max-w-[1320px] px-4 pb-16 pt-7 md:px-8 md:pt-10">
        <div className="mb-7 grid gap-5 md:mb-9 md:grid-cols-[1fr_auto] md:items-end">
          <div>
            <div className="mb-3 flex items-center gap-2 eyebrow"><span className="h-px w-6 bg-primary" /> Прототип · оценка риска банкротства</div>
            <h1 className="serif max-w-3xl text-[39px] leading-[1.02] tracking-[-.045em] md:text-[52px]">Модель показывает<br className="hidden sm:block" /> свою работу.</h1>
            <p className="mt-4 max-w-xl text-[12px] leading-relaxed text-muted-foreground md:text-[13px]">Исследуйте оценку вероятности и факторы, стоящие за ней. Все сведения о готовности и каждый прогноз поступают непосредственно из API.</p>
          </div>
          <div className="flex items-center gap-2 border-l-2 border-accent pl-3 text-[10px] leading-relaxed text-muted-foreground md:max-w-[250px]">
            <Info size={15} className="shrink-0 text-accent" />
            <span>Неофициальная техническая демонстрация. Не кредитное решение.</span>
          </div>
        </div>

        <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1.55fr)_minmax(270px,.75fr)]">
          <div className="space-y-5">
            <PredictionForm mode={mode} setMode={setMode} model={modelQuery.data} onResult={setResult} onPending={setInProgress} />
            {inProgress && !response && (
              <section className="panel p-5 md:p-7" aria-label="Загрузка результата" data-testid="loading-prediction">
                <div className="skeleton h-4 w-32 rounded-sm" /><div className="skeleton mt-4 h-14 w-56 rounded-sm" /><div className="skeleton mt-5 h-2 w-full rounded-sm" /><div className="skeleton mt-6 h-12 w-full rounded-sm" />
              </section>
            )}
            {response ? <ResultPanel result={response} /> : !inProgress && (
              <div className="panel flex min-h-[180px] flex-col items-center justify-center px-6 py-9 text-center" data-testid="empty-result">
                <div className="mb-3 flex h-10 w-10 items-center justify-center rounded-full border border-border text-primary"><Activity size={17} strokeWidth={1.5} /></div>
                <div className="text-[12px] font-bold">Здесь появится результат</div>
                <p className="mt-1.5 max-w-sm text-[10px] leading-relaxed text-muted-foreground">Выберите способ ввода и отправьте данные. Пока прогноз не запрошен, никакие значения не подставляются.</p>
              </div>
            )}
          </div>
          <aside className="space-y-4 lg:sticky lg:top-[92px]">
            <Readiness model={modelQuery.data} isLoading={modelQuery.isLoading} isError={modelQuery.isError} onRetry={() => { void modelQuery.refetch(); void healthQuery.refetch(); }} />
            <section className="border border-border bg-secondary/60 p-5">
              <div className="flex items-center gap-2"><CircleHelp size={14} className="text-primary" /><h2 className="text-[11px] font-bold">Как читать оценку</h2></div>
              <p className="mt-2 text-[10px] leading-relaxed text-muted-foreground">Вероятность — выход модели, а не фактическая вероятность события для конкретной компании. Вклады показывают связь признаков с выводом модели, не причинность.</p>
              <div className="mt-4 flex items-center gap-2 border-t border-border pt-3 mono text-[8px] uppercase tracking-[.08em] text-muted-foreground"><Clipboard size={12} /> Только для технического анализа</div>
            </section>
            <div className="flex items-start gap-2 px-1 text-[9px] leading-relaxed text-muted-foreground"><Info size={12} className="mt-0.5 shrink-0" /> Если API не сообщает готовность модели или данных, соответствующий способ прогноза останется недоступен.</div>
          </aside>
        </div>
        <footer className="mt-12 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4 text-[9px] text-muted-foreground">
          <span>КОНТУР РИСКА <span className="mx-1 opacity-50">/</span> ИССЛЕДОВАТЕЛЬСКИЙ СТЕНД</span>
          <span className="mono">ОЦЕНКА ≠ РЕШЕНИЕ</span>
        </footer>
      </main>
    </div>
  );
}

function Router() {
  return (
    <RoutedErrorBoundary>
      <Switch>
        <Route path="/" component={Home} />
        <Route component={NotFound} />
      </Switch>
    </RoutedErrorBoundary>
  );
}

function RoutedErrorBoundary({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  return <ErrorBoundary resetKey={location}>{children}</ErrorBoundary>;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
