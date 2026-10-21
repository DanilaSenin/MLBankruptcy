import { mkdir, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { Storage } from "@google-cloud/storage";

const MODEL_VERSION = "dup_txt-20261021-1811";
const MODEL_PREFIX = `ml-models/bankruptcy/${MODEL_VERSION}/artifacts`;

const storage = new Storage();

interface StorageLocation {
  bucketName: string;
  prefix: string;
}

function getStorageLocation(): StorageLocation {
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
    prefix: `${[basePrefix, MODEL_PREFIX].filter(Boolean).join("/")}/`,
  };
}

async function isFile(filePath: string): Promise<boolean> {
  try {
    return (await stat(filePath)).isFile();
  } catch {
    return false;
  }
}

export async function ensureModelArtifacts(
  localArtifactDir: string,
): Promise<string> {
  const configPath = path.join(localArtifactDir, "config.json");
  if (await isFile(configPath)) return localArtifactDir;

  const cacheParent = path.join(
    os.tmpdir(),
    `bankruptcy-model-${MODEL_VERSION}`,
  );
  const cacheDir = path.join(cacheParent, "artifacts");
  const completeMarker = path.join(cacheParent, ".complete");
  if (
    (await isFile(completeMarker)) &&
    (await isFile(path.join(cacheDir, "config.json")))
  ) {
    return cacheDir;
  }

  const { bucketName, prefix } = getStorageLocation();
  const bucket = storage.bucket(bucketName);
  const [files] = await bucket.getFiles({ prefix });
  if (files.length === 0) {
    throw new Error(
      `No private model assets found under ${prefix}. Upload the supplied model artifacts first.`,
    );
  }

  await mkdir(cacheDir, { recursive: true });
  for (const file of files) {
    const relativePath = file.name.slice(prefix.length);
    const segments = relativePath.split("/");
    if (!relativePath || segments.some((segment) => segment === "..")) {
      throw new Error("Model asset storage contains an invalid object path.");
    }

    const destination = path.join(cacheDir, ...segments);
    await mkdir(path.dirname(destination), { recursive: true });
    await file.download({ destination });
  }

  if (!(await isFile(path.join(cacheDir, "config.json")))) {
    throw new Error("Downloaded model assets are incomplete: config.json is missing.");
  }

  await writeFile(completeMarker, "ok\n", "utf8");
  return cacheDir;
}
