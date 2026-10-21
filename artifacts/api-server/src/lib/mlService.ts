import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { logger } from "./logger";
import { ensureModelArtifacts } from "./modelAssets";

interface MlInfo {
  model: string;
  version: string;
  oof_auc?: number | null;
  n_features: number;
  index_lookup: boolean;
  index_explanations?: boolean;
}

interface MlServiceStatus {
  ready: boolean;
  info: MlInfo | null;
  error: string | null;
}

export class MlServiceError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "MlServiceError";
  }
}

class MlServiceClient {
  private child: ChildProcess | null = null;
  private info: MlInfo | null = null;
  private lastError: string | null = null;
  private startPromise: Promise<void> | null = null;
  private readonly port = Number(process.env["ML_SERVICE_PORT"] ?? 8137);
  private readonly baseUrl = `http://127.0.0.1:${this.port}`;

  start(mlDirectory: string, workspaceRoot: string): Promise<void> {
    if (this.startPromise) return this.startPromise;
    this.startPromise = this.startInternal(mlDirectory, workspaceRoot);
    return this.startPromise;
  }

  status(): MlServiceStatus {
    const alive = this.child !== null && this.child.exitCode === null;
    return {
      ready: alive && this.info !== null,
      info: alive ? this.info : null,
      error: this.lastError,
    };
  }

  async predictByIndex(index: number): Promise<unknown> {
    return this.request(`/predict/${index}?top_k=5&n_words=10`);
  }

  async predictFromFeatures(body: {
    id?: number | null;
    features: Record<string, number | null>;
    text_features?: Record<string, number>;
  }): Promise<unknown> {
    const wordCount = Object.entries(body.text_features ?? {})
      .map(([word, count]) => `${word} ${count}`)
      .join(" ");

    return this.request("/predict?top_k=5&n_words=10", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: body.id ?? null,
        features: body.features,
        word_count: wordCount || null,
      }),
    });
  }

  async predictFromCsv(features: Record<string, number | null>): Promise<unknown> {
    return this.predictFromFeatures({ features });
  }

  stop(): void {
    this.child?.kill("SIGTERM");
  }

  private async startInternal(
    mlDirectory: string,
    workspaceRoot: string,
  ): Promise<void> {
    try {
      const artifacts = await ensureModelArtifacts(
        path.join(mlDirectory, "artifacts"),
      );
      const env: NodeJS.ProcessEnv = {
        ...process.env,
        ML_ARTIFACTS: artifacts,
      };
      const libgomp = this.findOpenMpLibrary();
      if (libgomp) {
        env["LD_LIBRARY_PATH"] = [
          path.dirname(libgomp),
          env["LD_LIBRARY_PATH"],
        ]
          .filter(Boolean)
          .join(path.delimiter);
      }

      this.child = spawn(
        "uv",
        [
          "run",
          "--project",
          workspaceRoot,
          "python",
          "-m",
          "uvicorn",
          "ml_service:app",
          "--host",
          "127.0.0.1",
          "--port",
          String(this.port),
          "--no-access-log",
        ],
        {
          cwd: mlDirectory,
          env,
          stdio: ["ignore", "pipe", "pipe"],
        },
      );

      this.child.stdout?.on("data", (chunk: Buffer) => {
        const message = chunk.toString().trim();
        if (message) logger.info({ message }, "ML service");
      });
      this.child.stderr?.on("data", (chunk: Buffer) => {
        const message = chunk.toString().trim();
        if (message) logger.info({ message }, "ML service");
      });
      this.child.once("error", (error) => {
        this.lastError = `Could not start the ML service: ${error.message}`;
        logger.error({ err: error }, "Could not start ML inference service");
      });
      this.child.once("exit", (code, signal) => {
        if (code !== 0) {
          this.lastError =
            this.lastError ??
            `ML service stopped unexpectedly (code ${String(code)}, signal ${String(signal)}).`;
          this.info = null;
          logger.error({ code, signal }, "ML inference service exited");
        }
      });

      await this.waitUntilReady();
    } catch (error) {
      this.lastError =
        error instanceof Error ? error.message : "ML service startup failed.";
      throw error;
    }
  }

  private findOpenMpLibrary(): string | null {
    try {
      const resolved = execFileSync("gcc", ["-print-file-name=libgomp.so.1"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      }).trim();
      return path.isAbsolute(resolved) ? resolved : null;
    } catch {
      return null;
    }
  }

  private async waitUntilReady(): Promise<void> {
    for (let attempt = 0; attempt < 90; attempt += 1) {
      if (this.child?.exitCode !== null && this.child?.exitCode !== undefined) {
        throw new Error(this.lastError ?? "ML service exited before becoming ready.");
      }

      try {
        const response = await fetch(`${this.baseUrl}/health`, {
          signal: AbortSignal.timeout(1500),
        });
        if (response.ok) {
          this.info = (await response.json()) as MlInfo;
          this.lastError = null;
          logger.info(
            { model: this.info.model, version: this.info.version },
            "ML inference service ready",
          );
          return;
        }
      } catch {
        // The model loads before the Python service can answer health checks.
      }

      await delay(1000);
    }

    throw new Error("ML service did not become ready within 90 seconds.");
  }

  private async request(endpoint: string, init?: RequestInit): Promise<unknown> {
    if (!this.status().ready) {
      throw new MlServiceError(
        this.lastError ?? "The prediction model is still starting or unavailable.",
        503,
      );
    }

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${endpoint}`, {
        ...init,
        signal: AbortSignal.timeout(30_000),
      });
    } catch {
      throw new MlServiceError("The prediction service could not be reached.", 503);
    }

    const body: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const detail =
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
          ? body.detail
          : "Prediction failed.";
      throw new MlServiceError(detail, response.status);
    }

    return body;
  }
}

export const mlService = new MlServiceClient();
