import { createReadStream } from "node:fs";
import { readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "node:stream/promises";
import { Storage } from "@google-cloud/storage";

const MODEL_VERSION = "dup_txt-20261021-1811";
const MODEL_PREFIX = `ml-models/bankruptcy/${MODEL_VERSION}/artifacts`;

const storage = new Storage();

function getStorageLocation(): { bucketName: string; prefix: string } {
  const modelAssetsUri = process.env["MODEL_ASSETS_GCS_URI"]?.trim();
  const match = modelAssetsUri?.match(/^gs:\/\/([^/]+)(?:\/(.*))?$/);
  if (!match) {
    throw new Error(
      "MODEL_ASSETS_GCS_URI must be a gs://bucket/optional-prefix URI.",
    );
  }

  const bucketName = match[1];
  const basePrefix = (match[2] ?? "").replace(/^\/+|\/+$/g, "");

  return {
    bucketName,
    prefix: [basePrefix, MODEL_PREFIX].filter(Boolean).join("/"),
  };
}

async function listFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const results: string[] = [];

  for (const entry of entries) {
    const absolute = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      results.push(...(await listFiles(absolute)));
    } else if (entry.isFile()) {
      results.push(absolute);
    }
  }

  return results;
}

async function main(): Promise<void> {
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const workspaceRoot = path.resolve(scriptDir, "../..");
  const artifactsDir = path.join(
    workspaceRoot,
    "artifacts/api-server/ml/artifacts",
  );
  const localFiles = await listFiles(artifactsDir);
  if (localFiles.length === 0) {
    throw new Error(`No model assets found in ${artifactsDir}.`);
  }

  const { bucketName, prefix } = getStorageLocation();
  const bucket = storage.bucket(bucketName);

  for (const absolutePath of localFiles) {
    const relativePath = path
      .relative(artifactsDir, absolutePath)
      .split(path.sep)
      .join("/");
    const objectName = `${prefix}/${relativePath}`;
    await pipeline(
      createReadStream(absolutePath),
      bucket.file(objectName).createWriteStream({
        resumable: false,
        metadata: {
          cacheControl: "private, no-store",
          contentType: "application/octet-stream",
        },
      }),
    );
    process.stdout.write(`Uploaded ${relativePath}\n`);
  }

  process.stdout.write(
    `Uploaded ${localFiles.length} model files to Google Cloud Storage.\n`,
  );
}

main().catch((error: unknown) => {
  process.stderr.write(
    `${error instanceof Error ? error.message : String(error)}\n`,
  );
  process.exitCode = 1;
});
