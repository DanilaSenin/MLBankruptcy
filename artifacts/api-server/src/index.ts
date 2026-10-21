import app from "./app";
import { logger } from "./lib/logger";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { mlService } from "./lib/mlService";

const rawPort = process.env["PORT"];

if (!rawPort) {
  throw new Error(
    "PORT environment variable is required but was not provided.",
  );
}

const port = Number(rawPort);
const distDirectory = path.dirname(fileURLToPath(import.meta.url));
const apiDirectory = path.resolve(distDirectory, "..");
const mlDirectory = path.join(apiDirectory, "ml");
const workspaceRoot = path.resolve(apiDirectory, "../..");

if (Number.isNaN(port) || port <= 0) {
  throw new Error(`Invalid PORT value: "${rawPort}"`);
}

const server = app.listen(port, (err) => {
  if (err) {
    logger.error({ err }, "Error listening on port");
    process.exit(1);
  }

  logger.info({ port }, "Server listening");
});

void mlService.start(mlDirectory, workspaceRoot).catch((error: unknown) => {
  logger.error({ err: error }, "ML inference service unavailable");
});

function shutdown(signal: string): void {
  logger.info({ signal }, "Shutting down API server");
  mlService.stop();
  server.close(() => process.exit(0));
}

process.once("SIGINT", () => shutdown("SIGINT"));
process.once("SIGTERM", () => shutdown("SIGTERM"));
