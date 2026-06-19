import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";

export type ProviderType = "gemini" | "groq" | "openai-compatible";
export type AccessPolicy = "self_only" | "whitelist" | "everyone_except_blacklist";

export interface AgentPolicy {
  autoReplyEnabled: boolean;
  readPolicy: AccessPolicy;
  replyPolicy: AccessPolicy;
  sendPolicy: AccessPolicy;
  allowGroups: boolean;
  whitelistedContacts: string[];
  blacklistedContacts: string[];
  whitelistedGroups: string[];
  mentionBypass: string;
}

export interface AgentState {
  mode: "bootstrap" | "configured";
  bootstrapStarted: boolean;
  pendingSystemPrompt?: string;
  systemPrompt?: string;
  memorySummary?: string;
  bootstrapNotes: string[];
  policy: AgentPolicy;
}

export interface WagentConfig {
  provider: ProviderType;
  apiKey: string;
  baseUrl?: string;
  model: string;
  fallbackModels: string[];
  replyDelayMinMs: number;
  replyDelayMaxMs: number;
  agent: AgentState;
}

export function getConfigDir(): string {
  if (process.env.WAGENT_HOME) return process.env.WAGENT_HOME;
  if (process.platform === "win32") return join(process.env.APPDATA ?? homedir(), "wagent");
  return join(process.env.XDG_CONFIG_HOME ?? join(homedir(), ".config"), "wagent");
}

export function ensureConfigDir(): string {
  const dir = getConfigDir();
  mkdirSync(dir, { recursive: true });
  return dir;
}

export function getConfigPath(): string {
  return join(ensureConfigDir(), "config.json");
}

export function defaultPolicy(): AgentPolicy {
  return {
    autoReplyEnabled: false,
    readPolicy: "self_only",
    replyPolicy: "self_only",
    sendPolicy: "everyone_except_blacklist",
    allowGroups: false,
    whitelistedContacts: [],
    blacklistedContacts: [],
    whitelistedGroups: [],
    mentionBypass: "@wagent",
  };
}

export function defaultAgentState(): AgentState {
  return {
    mode: "bootstrap",
    bootstrapStarted: false,
    bootstrapNotes: [],
    policy: defaultPolicy(),
  };
}

export function defaultConfig(): WagentConfig {
  return {
    provider: "gemini",
    apiKey: "",
    model: "gemini-3.1-flash-lite",
    fallbackModels: ["gemma-4-31b", "gemma-4-27b"],
    replyDelayMinMs: 2000,
    replyDelayMaxMs: 8000,
    agent: defaultAgentState(),
  };
}

export function loadConfig(): WagentConfig {
  const path = getConfigPath();
  const fromDisk = existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : {};
  const merged: WagentConfig = {
    ...defaultConfig(),
    ...fromDisk,
    agent: {
      ...defaultAgentState(),
      ...(fromDisk.agent ?? {}),
      policy: { ...defaultPolicy(), ...(fromDisk.agent?.policy ?? {}) },
      bootstrapNotes: fromDisk.agent?.bootstrapNotes ?? [],
    },
  };

  if (process.env.GROQ_API_KEY) {
    merged.provider = "groq";
    merged.apiKey = process.env.GROQ_API_KEY;
    merged.baseUrl = process.env.GROQ_BASE_URL ?? merged.baseUrl;
    merged.model = process.env.GROQ_MODEL ?? "openai/gpt-oss-120b";
    merged.fallbackModels = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"];
  }

  return merged;
}

export function saveConfig(config: WagentConfig): void {
  writeFileSync(getConfigPath(), `${JSON.stringify(config, null, 2)}\n`, { mode: 0o600 });
}

function mask(value: string): string {
  if (value.length <= 8) return "********";
  return `${value.slice(0, 4)}...${value.slice(-4)}`;
}

export async function ensureAiConfig(config: WagentConfig): Promise<WagentConfig> {
  if (config.apiKey) return config;

  const rl = readline.createInterface({ input, output });
  try {
    console.log("\nChoose AI provider:");
    console.log("1. Google Gemini + Web Search (recommended)");
    console.log("2. Groq");
    console.log("3. Other OpenAI-compatible endpoint");
    const choice = (await rl.question("Provider [1]: ")).trim() || "1";

    if (choice === "2") {
      config.provider = "groq";
      config.baseUrl = "https://api.groq.com/openai/v1";
      config.model = "openai/gpt-oss-120b";
      config.fallbackModels = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"];
    } else if (choice === "3") {
      config.provider = "openai-compatible";
      config.baseUrl = (await rl.question("Base URL: ")).trim();
      config.model = (await rl.question("Model: ")).trim();
      config.fallbackModels = [];
    } else {
      config.provider = "gemini";
      config.model = "gemini-3.1-flash-lite";
      config.fallbackModels = ["gemma-4-31b", "gemma-4-27b"];
    }

    config.apiKey = (await rl.question("API key: ")).trim();
    if (!config.apiKey) throw new Error("API key is required");
    saveConfig(config);
    console.log(`Saved ${config.provider} config (${mask(config.apiKey)}).`);
    return config;
  } finally {
    rl.close();
  }
}
