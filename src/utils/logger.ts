// ============================================================
// WA MCP — Structured Logger with Request Tracing
// ============================================================

import pino from "pino";
import { SERVER_NAME, DEFAULT_LOG_LEVEL } from "../constants.js";

const logLevel = process.argv.includes("--debug")
  ? "debug"
  : process.env.LOG_LEVEL
    ? process.env.LOG_LEVEL
    : "info";
const isStdio = process.env.WA_TRANSPORT === "stdio";

export const logger = pino(
  {
    name: SERVER_NAME,
    level: logLevel,
    transport:
      process.env.NODE_ENV !== "production"
        ? { target: "pino-pretty", options: { colorize: true, destination: isStdio ? 2 : 1 } }
        : undefined,
    redact: [
      "headers.authorization",
      "*.accessToken",
      "*.cloudAccessToken",
      "*.cloud_access_token",
      "*.apiKey",
      "params.accessToken",
      "params.apiKey",
      "req.headers.authorization",
      "req.headers['x-api-key']",
    ],
  },
  pino.destination(isStdio ? 2 : 1),
);

export type Logger = pino.Logger;

export function createChildLogger(context: Record<string, unknown>): pino.Logger {
  return logger.child(context);
}

export function createRequestLogger(toolName: string, instanceId?: string): pino.Logger {
  return logger.child({
    requestId: crypto.randomUUID(),
    tool: toolName,
    ...(instanceId ? { instanceId } : {}),
  });
}
