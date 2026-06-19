#!/usr/bin/env node
import "./quiet-console.js";
import { InstanceManager } from "./services/instance-manager.js";
import { BaileysAdapter } from "./channels/baileys/baileys.adapter.js";
import { createChildLogger } from "./utils/logger.js";
import { generateText, tool } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { z } from "zod";
import qrcode from "qrcode-terminal";
import { ensureAiConfig, loadConfig, saveConfig, type WagentConfig } from "./config.js";
import type {
  NormalizedMessageEvent,
  NormalizedConnectionEvent,
  NormalizedPresenceEvent,
  MessageContent,
} from "./types/channel.types.js";
const logger = createChildLogger({ service: "wagent" });

const chatHistories = new Map<string, Array<{ role: "user" | "assistant"; content: string }>>();
const MAX_HISTORY = 30;
const processedMessages = new Set<string>();
const agentSentMessages = new Set<string>();
const typingUntil = new Map<string, number>();

function getOwnJid(adapter: BaileysAdapter): string | null {
  return adapter.getMyJid();
}

function toUserJid(jid: string | null): string | null {
  if (!jid || !jid.includes("@")) return jid;
  const [user, domain] = jid.split("@");
  return `${user.split(":")[0]}@${domain}`;
}

function getContactName(adapter: BaileysAdapter, jid: string): Promise<string | null> {
  return adapter.getContacts(jid).then((c) => {
    const match = c.find((x) => x.jid === jid);
    return match?.name ?? match?.notifyName ?? jid.split("@")[0] ?? null;
  });
}

function getGroupName(adapter: BaileysAdapter, jid: string): Promise<string> {
  return adapter.getGroupMetadata(jid).then((g) => g.subject).catch(() => jid.split("@")[0]);
}

type ChatCtx = { currentChatId: string; config: WagentConfig };

function getModelNames(config: WagentConfig): string[] {
  return [config.model, ...config.fallbackModels].filter(Boolean);
}

async function getLanguageModel(config: WagentConfig, modelName: string): Promise<any> {
  if (config.provider === "gemini") {
    const mod = await import("@ai-sdk/google");
    const createGoogleGenerativeAI = (mod as any).createGoogleGenerativeAI;
    const google = createGoogleGenerativeAI({ apiKey: config.apiKey });
    return google(modelName);
  }

  const provider = createOpenAICompatible({
    name: config.provider,
    baseURL: config.baseUrl ?? "https://api.groq.com/openai/v1",
    apiKey: config.apiKey,
  });
  return provider.chatModel(modelName);
}

async function generateWithFallback(config: WagentConfig, options: Record<string, unknown>) {
  let lastError: unknown;
  for (const modelName of getModelNames(config)) {
    try {
      return await generateText({ ...(options as any), model: await getLanguageModel(config, modelName) });
    } catch (err) {
      lastError = err;
      logger.warn(`Fallback: ${modelName} failed`);
    }
  }
  throw lastError;
}

function randomBetween(min: number, max: number): number {
  return Math.floor(min + Math.random() * (max - min + 1));
}

async function delayBeforeReply(config: WagentConfig): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, randomBetween(config.replyDelayMinMs, config.replyDelayMaxMs)));
}

async function minDelayBeforeReply(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, randomBetween(500, 1500)));
}

async function waitForTypingToStop(chatId: string): Promise<void> {
  const started = Date.now();
  while ((typingUntil.get(chatId) ?? 0) > Date.now() && Date.now() - started < 30_000) {
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
}

function rememberAgentMessage(messageId: string): void {
  if (!messageId) return;
  agentSentMessages.add(messageId);
  if (agentSentMessages.size > 1000) {
    const toDelete = [...agentSentMessages].slice(0, 500);
    for (const id of toDelete) agentSentMessages.delete(id);
  }
}

function normalizePhone(input: string): string | null {
  const digits = input.replace(/[^0-9]/g, "");
  return digits.length >= 8 ? `${digits}@s.whatsapp.net` : null;
}

function normalizeName(input: string): string {
  return input.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function editDistance(a: string, b: string): number {
  const dp = Array.from({ length: a.length + 1 }, (_, i) =>
    Array.from({ length: b.length + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0)),
  );
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
    }
  }
  return dp[a.length][b.length];
}

async function resolveTarget(adapter: BaileysAdapter, to: string | undefined, ctx: ChatCtx): Promise<string> {
  const raw = to?.trim();
  if (!raw) return ctx.currentChatId;
  if (raw.includes("@")) {
    if (raw.startsWith("@")) throw new Error(`Invalid target JID: ${raw}`);
    return raw;
  }

  const phoneJid = normalizePhone(raw);
  if (phoneJid) return phoneJid;

  let contacts = await adapter.getContacts(raw);
  if (contacts.length === 0) {
    const needle = normalizeName(raw);
    const fuzzy = (await adapter.getContacts())
      .map((contact) => {
        const names = [contact.name, contact.notifyName].filter(Boolean).map((name) => normalizeName(name!));
        const distance = Math.min(...names.map((name) => editDistance(needle, name)), Number.POSITIVE_INFINITY);
        return { contact, distance };
      })
      .filter(({ distance }) => distance <= 2)
      .sort((a, b) => a.distance - b.distance);
    if (fuzzy.length > 0) contacts = [fuzzy[0].contact];
  }
  const query = raw.toLowerCase();
  const exact = contacts.find((c) =>
    [c.name, c.notifyName, c.phone]
      .filter(Boolean)
      .some((v) => v!.toLowerCase() === query),
  );
  const match = exact ?? contacts[0];
  if (!match) throw new Error(`Could not resolve contact "${raw}". Try a phone number or exact contact name.`);
  if (contacts.length > 1 && !exact) {
    const names = contacts.slice(0, 5).map((c) => c.name ?? c.notifyName ?? c.phone ?? c.jid).join(", ");
    throw new Error(`Multiple contacts match "${raw}": ${names}. Use a more specific name or phone number.`);
  }
  const resolved = match.phone ? `${match.phone}@s.whatsapp.net` : match.jid;
  if (!resolved || resolved.startsWith("@")) throw new Error(`Resolved invalid target for "${raw}": ${resolved}`);
  return resolved;
}

function buildTools(adapter: BaileysAdapter) {
  return {
    sendText: tool({
      description: "Send a text message to a phone number or JID. Omit 'to' to send to the current conversation.",
      inputSchema: z.object({
        to: z.string().optional().describe("Phone/JID. Defaults to current chat."),
        text: z.string().describe("Message content"),
        quotedMessageId: z.string().optional().describe("Reply to a specific message ID"),
      }),
      execute: async ({ to, text, quotedMessageId }, { experimental_context }: any) => {
        const ctx = experimental_context as ChatCtx;
        const target = await resolveTarget(adapter, to, ctx);
        const content: MessageContent = { type: "text", text: withAgentPrefix(text) };
        if (quotedMessageId) content.quotedMessageId = quotedMessageId;
        await minDelayBeforeReply();
        const res = await adapter.sendMessage(target, content);
        rememberAgentMessage(res.messageId);
        return `Sent to ${target}. ID: ${res.messageId}`;
      },
    }),

    sendImage: tool({
      description: "Send an image from a public HTTPS URL. Omit 'to' to send to the current conversation.",
      inputSchema: z.object({
        to: z.string().optional().describe("Phone/JID. Defaults to current chat."),
        imageUrl: z.string().url().describe("Public HTTPS URL of the image"),
        caption: z.string().optional(),
      }),
      execute: async ({ to, imageUrl, caption }, { experimental_context }: any) => {
        const ctx = experimental_context as ChatCtx;
        const target = await resolveTarget(adapter, to, ctx);
        await minDelayBeforeReply();
        const res = await adapter.sendMessage(target, { type: "image", image: imageUrl, caption });
        rememberAgentMessage(res.messageId);
        return `Image sent. ID: ${res.messageId}`;
      },
    }),

    sendVideo: tool({
      description: "Send a video from a public HTTPS URL. Omit 'to' to send to the current conversation.",
      inputSchema: z.object({
        to: z.string().optional().describe("Phone/JID. Defaults to current chat."),
        videoUrl: z.string().url(),
        caption: z.string().optional(),
      }),
      execute: async ({ to, videoUrl, caption }, { experimental_context }: any) => {
        const ctx = experimental_context as ChatCtx;
        const target = await resolveTarget(adapter, to, ctx);
        await minDelayBeforeReply();
        const res = await adapter.sendMessage(target, { type: "video", video: videoUrl, caption });
        rememberAgentMessage(res.messageId);
        return `Video sent. ID: ${res.messageId}`;
      },
    }),

    sendAudio: tool({
      description: "Send an audio file or voice note from a URL. Omit 'to' to send to the current conversation.",
      inputSchema: z.object({
        to: z.string().optional().describe("Phone/JID. Defaults to current chat."),
        audioUrl: z.string().url(),
        asVoiceNote: z.boolean().optional().default(false),
      }),
      execute: async ({ to, audioUrl, asVoiceNote }, { experimental_context }: any) => {
        const ctx = experimental_context as ChatCtx;
        const target = await resolveTarget(adapter, to, ctx);
        await minDelayBeforeReply();
        const res = await adapter.sendMessage(target, { type: "audio", audio: audioUrl, ptt: asVoiceNote });
        rememberAgentMessage(res.messageId);
        return `Audio sent. ID: ${res.messageId}`;
      },
    }),

    sendDocument: tool({
      description: "Send a document/file from a URL. Omit 'to' to send to the current conversation.",
      inputSchema: z.object({
        to: z.string().optional().describe("Phone/JID. Defaults to current chat."),
        documentUrl: z.string().url(),
        fileName: z.string().describe("Display filename e.g. report.pdf"),
        mimeType: z.string().describe("MIME type e.g. application/pdf"),
      }),
      execute: async ({ to, documentUrl, fileName, mimeType }, { experimental_context }: any) => {
        const ctx = experimental_context as ChatCtx;
        const target = await resolveTarget(adapter, to, ctx);
        await minDelayBeforeReply();
        const res = await adapter.sendMessage(target, { type: "document", document: documentUrl, fileName, mimeType });
        rememberAgentMessage(res.messageId);
        return `Document sent. ID: ${res.messageId}`;
      },
    }),

    sendLocation: tool({
      description: "Send a GPS location pin. Omit 'to' to send to the current conversation.",
      inputSchema: z.object({
        to: z.string().optional().describe("Phone/JID. Defaults to current chat."),
        latitude: z.number(),
        longitude: z.number(),
        name: z.string().optional(),
        address: z.string().optional(),
      }),
      execute: async ({ to, latitude, longitude, name, address }, { experimental_context }: any) => {
        const ctx = experimental_context as ChatCtx;
        const target = await resolveTarget(adapter, to, ctx);
        await minDelayBeforeReply();
        const res = await adapter.sendMessage(target, { type: "location", latitude, longitude, name, address });
        rememberAgentMessage(res.messageId);
        return `Location sent. ID: ${res.messageId}`;
      },
    }),

    sendContact: tool({
      description: "Send a contact (vCard). Omit 'to' to send to the current conversation.",
      inputSchema: z.object({
        to: z.string().optional().describe("Phone/JID. Defaults to current chat."),
        contactName: z.string(),
        contactPhone: z.string(),
      }),
      execute: async ({ to, contactName, contactPhone }, { experimental_context }: any) => {
        const ctx = experimental_context as ChatCtx;
        const target = await resolveTarget(adapter, to, ctx);
        await minDelayBeforeReply();
        const res = await adapter.sendMessage(target, { type: "contact", contactName, contactPhone });
        rememberAgentMessage(res.messageId);
        return `Contact sent. ID: ${res.messageId}`;
      },
    }),

    sendPoll: tool({
      description: "Create and send a poll to a group or chat. Omit 'to' to send to the current conversation.",
      inputSchema: z.object({
        to: z.string().optional().describe("Phone/JID. Defaults to current chat."),
        question: z.string(),
        options: z.array(z.string()).min(1).max(12),
        multiSelect: z.boolean().optional().default(false),
      }),
      execute: async ({ to, question, options, multiSelect }, { experimental_context }: any) => {
        const ctx = experimental_context as ChatCtx;
        const target = await resolveTarget(adapter, to, ctx);
        await minDelayBeforeReply();
        const res = await adapter.sendMessage(target, { type: "poll", question, options, multiSelect });
        rememberAgentMessage(res.messageId);
        return `Poll sent. ID: ${res.messageId}`;
      },
    }),

    sendReaction: tool({
      description: "React to a message with an emoji",
      inputSchema: z.object({
        chatId: z.string(),
        messageId: z.string(),
        emoji: z.string().describe("Single emoji character"),
      }),
      execute: async ({ chatId, messageId, emoji }) => {
        await adapter.sendReaction(chatId, messageId, emoji);
        return `Reacted with ${emoji}`;
      },
    }),

    forwardMessage: tool({
      description: "Forward a message from one chat to another",
      inputSchema: z.object({
        to: z.string(),
        messageId: z.string(),
        fromChatId: z.string(),
      }),
      execute: async ({ to, messageId, fromChatId }) => {
        const res = await adapter.forwardMessage(to, messageId, fromChatId);
        return `Forwarded. ID: ${res.messageId}`;
      },
    }),

    editMessage: tool({
      description: "Edit a message you sent (works on messages less than ~48h old)",
      inputSchema: z.object({
        chatId: z.string(),
        messageId: z.string(),
        newText: z.string(),
      }),
      execute: async ({ chatId, messageId, newText }) => {
        await adapter.editMessage(chatId, messageId, newText);
        return "Message edited";
      },
    }),

    deleteMessage: tool({
      description: "Delete a message for everyone",
      inputSchema: z.object({
        chatId: z.string(),
        messageId: z.string(),
      }),
      execute: async ({ chatId, messageId }) => {
        await adapter.deleteMessage(chatId, messageId);
        return "Message deleted";
      },
    }),

    pinMessage: tool({
      description: "Pin or unpin a message in a chat",
      inputSchema: z.object({
        chatId: z.string(),
        messageId: z.string(),
        pin: z.boolean(),
      }),
      execute: async ({ chatId, messageId, pin }) => {
        await adapter.pinMessage(chatId, messageId, pin);
        return pin ? "Message pinned" : "Message unpinned";
      },
    }),

    sendPresence: tool({
      description: "Show typing indicator or recording indicator in a chat",
      inputSchema: z.object({
        chatId: z.string(),
        status: z.enum(["composing", "recording", "paused", "available", "unavailable"]),
      }),
      execute: async ({ chatId, status }) => {
        await adapter.sendPresence(chatId, status);
        return `Presence: ${status}`;
      },
    }),

    markRead: tool({
      description: "Mark messages as read in a chat",
      inputSchema: z.object({
        chatId: z.string(),
        messageIds: z.array(z.string()),
      }),
      execute: async ({ chatId, messageIds }) => {
        await adapter.markRead(chatId, messageIds);
        return "Marked as read";
      },
    }),

    getMessages: tool({
      description: "Get recent messages from a chat or group",
      inputSchema: z.object({
        chatId: z.string(),
        limit: z.number().min(1).max(100).default(20),
      }),
      execute: async ({ chatId, limit }) => {
        const msgs = await adapter.getMessages(chatId, limit);
        return JSON.stringify(msgs.map((m) => ({
          from: m.isFromMe ? "me" : m.senderId,
          text: m.content,
          time: new Date(m.timestamp * 1000).toISOString(),
          type: m.type,
        })));
      },
    }),

    searchContact: tool({
      description: "Search contacts by name or phone number",
      inputSchema: z.object({
        query: z.string().describe("Name or partial phone number"),
      }),
      execute: async ({ query }) => {
        const contacts = await adapter.getContacts(query);
        return JSON.stringify(contacts.map((c) => ({
          jid: c.jid,
          name: c.name ?? c.notifyName,
          phone: c.phone,
          isBusiness: c.isBusiness,
        })));
      },
    }),

    checkNumber: tool({
      description: "Check if a phone number exists on WhatsApp",
      inputSchema: z.object({
        phone: z.string().describe("Phone number with country code"),
      }),
      execute: async ({ phone }) => {
        const result = await adapter.checkNumberExists(phone);
        return JSON.stringify(result);
      },
    }),

    blockContact: tool({
      description: "Block a contact by JID",
      inputSchema: z.object({ jid: z.string() }),
      execute: async ({ jid }) => {
        await adapter.blockContact(jid);
        return "Contact blocked";
      },
    }),

    unblockContact: tool({
      description: "Unblock a contact by JID",
      inputSchema: z.object({ jid: z.string() }),
      execute: async ({ jid }) => {
        await adapter.unblockContact(jid);
        return "Contact unblocked";
      },
    }),

    createGroup: tool({
      description: "Create a new WhatsApp group",
      inputSchema: z.object({
        name: z.string(),
        participants: z.array(z.string()).describe("Array of phone numbers or JIDs to add"),
      }),
      execute: async ({ name, participants }) => {
        const res = await adapter.createGroup(name, participants);
        return `Group created. ID: ${res.groupId}`;
      },
    }),

    groupAddParticipants: tool({
      description: "Add participants to a group",
      inputSchema: z.object({
        groupId: z.string(),
        participants: z.array(z.string()),
      }),
      execute: async ({ groupId, participants }) => {
        await adapter.modifyParticipants(groupId, participants, "add");
        return "Participants added";
      },
    }),

    groupRemoveParticipants: tool({
      description: "Remove participants from a group",
      inputSchema: z.object({
        groupId: z.string(),
        participants: z.array(z.string()),
      }),
      execute: async ({ groupId, participants }) => {
        await adapter.modifyParticipants(groupId, participants, "remove");
        return "Participants removed";
      },
    }),

    groupPromote: tool({
      description: "Promote participants to admin in a group",
      inputSchema: z.object({
        groupId: z.string(),
        participants: z.array(z.string()),
      }),
      execute: async ({ groupId, participants }) => {
        await adapter.modifyParticipants(groupId, participants, "promote");
        return "Participants promoted";
      },
    }),

    groupDemote: tool({
      description: "Demote participants from admin in a group",
      inputSchema: z.object({
        groupId: z.string(),
        participants: z.array(z.string()),
      }),
      execute: async ({ groupId, participants }) => {
        await adapter.modifyParticipants(groupId, participants, "demote");
        return "Participants demoted";
      },
    }),

    groupUpdateSubject: tool({
      description: "Change group name/subject",
      inputSchema: z.object({
        groupId: z.string(),
        subject: z.string(),
      }),
      execute: async ({ groupId, subject }) => {
        await adapter.modifyGroup(groupId, { action: "updateSubject", value: subject });
        return "Group name updated";
      },
    }),

    groupUpdateDescription: tool({
      description: "Change group description",
      inputSchema: z.object({
        groupId: z.string(),
        description: z.string(),
      }),
      execute: async ({ groupId, description }) => {
        await adapter.modifyGroup(groupId, { action: "updateDescription", value: description });
        return "Group description updated";
      },
    }),

    groupUpdateSettings: tool({
      description: "Toggle between 'announcement' (admins only) and 'all' (everyone) for group messaging",
      inputSchema: z.object({
        groupId: z.string(),
        mode: z.enum(["announcement", "all"]),
      }),
      execute: async ({ groupId, mode }) => {
        await adapter.modifyGroup(groupId, { action: "updateSettings", value: mode });
        return `Group settings updated to ${mode}`;
      },
    }),

    groupLeave: tool({
      description: "Leave a WhatsApp group",
      inputSchema: z.object({ groupId: z.string() }),
      execute: async ({ groupId }) => {
        await adapter.modifyGroup(groupId, { action: "leave" });
        return "Left the group";
      },
    }),

    groupGetInviteCode: tool({
      description: "Get the invite link/code for a group",
      inputSchema: z.object({ groupId: z.string() }),
      execute: async ({ groupId }) => {
        const code = await adapter.getGroupInviteCode(groupId);
        return `Invite code: ${code}`;
      },
    }),

    groupJoin: tool({
      description: "Join a group using an invite code",
      inputSchema: z.object({ inviteCode: z.string() }),
      execute: async ({ inviteCode }) => {
        const gid = await adapter.joinGroup(inviteCode);
        return `Joined group: ${gid}`;
      },
    }),

    getGroupMetadata: tool({
      description: "Get detailed metadata about a group (members, admins, settings)",
      inputSchema: z.object({ groupId: z.string() }),
      execute: async ({ groupId }) => {
        const meta = await adapter.getGroupMetadata(groupId);
        return JSON.stringify({
          subject: meta.subject,
          description: meta.description,
          participantCount: meta.participantCount,
          isAnnounce: meta.isAnnounce,
          admins: meta.participants.filter((p) => p.isAdmin).map((p) => p.jid),
        });
      },
    }),

    updateProfileName: tool({
      description: "Change the WhatsApp profile/display name",
      inputSchema: z.object({ name: z.string() }),
      execute: async ({ name }) => {
        await adapter.updateProfileName(name);
        return `Profile name changed to: ${name}`;
      },
    }),

    updateProfileStatus: tool({
      description: "Change the WhatsApp profile status/about text",
      inputSchema: z.object({ status: z.string() }),
      execute: async ({ status }) => {
        await adapter.updateProfileStatus(status);
        return "Profile status updated";
      },
    }),

    getProfileInfo: tool({
      description: "Get the logged-in user's profile info",
      inputSchema: z.object({}),
      execute: async () => {
        const info = await adapter.getProfileInfo();
        return JSON.stringify(info);
      },
    }),

    sendTextStatus: tool({
      description: "Post a text status/story update",
      inputSchema: z.object({
        text: z.string(),
        backgroundColor: z.string().optional().default("#000000"),
        font: z.number().optional().default(1),
      }),
      execute: async ({ text, backgroundColor, font }) => {
        await adapter.sendStatus({ type: "text", text, backgroundColor, font });
        return "Status posted";
      },
    }),

    sendImageStatus: tool({
      description: "Post an image as status/story update",
      inputSchema: z.object({
        imageUrl: z.string().url(),
        caption: z.string().optional(),
      }),
      execute: async ({ imageUrl, caption }) => {
        await adapter.sendStatus({ type: "image", media: imageUrl, caption });
        return "Image status posted";
      },
    }),

    archiveChat: tool({
      description: "Archive or unarchive a chat",
      inputSchema: z.object({
        chatId: z.string(),
        archive: z.boolean(),
      }),
      execute: async ({ chatId, archive }) => {
        await adapter.modifyChat(chatId, { action: archive ? "archive" : "unarchive" });
        return archive ? "Chat archived" : "Chat unarchived";
      },
    }),

    pinChat: tool({
      description: "Pin or unpin a chat",
      inputSchema: z.object({
        chatId: z.string(),
        pin: z.boolean(),
      }),
      execute: async ({ chatId, pin }) => {
        await adapter.modifyChat(chatId, { action: pin ? "pin" : "unpin" });
        return pin ? "Chat pinned" : "Chat unpinned";
      },
    }),

    muteChat: tool({
      description: "Mute or unmute a chat",
      inputSchema: z.object({
        chatId: z.string(),
        mute: z.boolean(),
      }),
      execute: async ({ chatId, mute }) => {
        await adapter.modifyChat(chatId, { action: mute ? "mute" : "unmute" });
        return mute ? "Chat muted" : "Chat unmuted";
      },
    }),
  };
}

const SYSTEM_PROMPT = `You are a WhatsApp assistant called "wagent". You have full access to all WhatsApp features.

CAPABILITIES:
- Send and receive text, images, videos, audio, documents, locations, contacts, polls
- Create and manage groups (add/remove/promote/demote members, change settings)
- Search contacts, check if numbers are on WhatsApp
- Forward, edit, delete, pin messages
- React to messages with emojis
- Show typing indicators and mark messages as read
- Manage profile (name, status)
- Post status/story updates
- Archive, pin, mute chats

CRITICAL RULES:
- Your response TEXT is automatically sent back to the conversation. NEVER call sendText/sendImage/etc to reply in the current conversation. Only use send tools to message OTHER people.
- Send tools accept contact names, phone numbers, or JIDs. If the user says "message Bharghav", call sendText with to="Bharghav"; the app resolves it safely.
- Use searchContact only when the user asks to look up contacts or when you need to disambiguate before answering.

BEHAVIOR:
- When the sender is YOU (self-message): treat it as a command. Execute the requested actions.
- When the sender is someone else: respond helpfully based on context.
- Unless instructed "silently", summarize what you did in a brief text response.
- You can chain multiple tool calls in sequence.
- Keep responses concise. Be direct and practical.`;

function getSystemPrompt(config: WagentConfig): string {
  const identity = config.agent.systemPrompt ? `\n\nAGENT-WRITTEN IDENTITY:\n${config.agent.systemPrompt}` : "";
  const memory = config.agent.memorySummary ? `\n\nCOMPACTED MEMORY:\n${config.agent.memorySummary}` : "";
  return `${SYSTEM_PROMPT}${identity}${memory}`;
}

function hasMentionBypass(text: string, config: WagentConfig): boolean {
  return text.toLowerCase().includes(config.agent.policy.mentionBypass.toLowerCase());
}

function isListed(value: string, list: string[]): boolean {
  const lower = value.toLowerCase();
  return list.some((item) => lower.includes(item.toLowerCase()) || item.toLowerCase() === lower);
}

function messageCanReachAgent(config: WagentConfig, event: NormalizedMessageEvent, isSelf: boolean, text: string): boolean {
  if (hasMentionBypass(text, config)) return true;
  if (isSelf) return true;
  if (config.agent.mode === "bootstrap") return false;
  if (!config.agent.policy.autoReplyEnabled) return false;

  const isGroup = event.chatId.endsWith("@g.us");
  if (isGroup && !config.agent.policy.allowGroups && !config.agent.policy.whitelistedGroups.includes(event.chatId)) return false;

  const sender = event.message.sender;
  if (isListed(sender, config.agent.policy.blacklistedContacts)) return false;
  if (config.agent.policy.readPolicy === "everyone_except_blacklist") return true;
  if (config.agent.policy.readPolicy === "whitelist") return isListed(sender, config.agent.policy.whitelistedContacts);
  return false;
}

function isConfirmation(text: string): boolean {
  return /\b(confirm|confirmed|approve|approved|yes|ship it|looks good)\b/i.test(text);
}

function isPromptDraftRequest(text: string): boolean {
  return /\b(enough|draft|write.*prompt|make.*prompt|system prompt|ready|done|that's it|that is it)\b/i.test(text);
}

function maybeExtractSayCommand(text: string): string | null {
  const match = text.match(/^\s*say\s+["']?(.+?)["']?\s*$/i);
  return match?.[1]?.trim() ?? null;
}

function withAgentPrefix(text: string): string {
  return text.trim().toLowerCase().startsWith("wagent:") ? text : `wagent: ${text}`;
}

async function sendTracked(adapter: BaileysAdapter, config: WagentConfig, to: string, text: string): Promise<void> {
  await delayBeforeReply(config);
  const res = await adapter.sendMessage(to, { type: "text", text: withAgentPrefix(text) });
  rememberAgentMessage(res.messageId);
}

async function startBootstrap(adapter: BaileysAdapter, config: WagentConfig, myJid: string): Promise<WagentConfig> {
  if (config.agent.bootstrapStarted || config.agent.mode !== "bootstrap") return config;
  logger.info("Bootstrap started");
  await sendTracked(adapter, config, myJid, "I'm online. I don't know who I am yet.");
  await delayBeforeReply(config);
  await sendTracked(adapter, config, myJid, "I need to know my purpose, what to call you, my vibe, and who I can read/reply to. Say 'draft prompt' when you're ready to lock it in.");
  config.agent.bootstrapStarted = true;
  saveConfig(config);
  return config;
}

async function handleBootstrapCommand(
  adapter: BaileysAdapter,
  config: WagentConfig,
  chatId: string,
  text: string,
): Promise<{ handled: true; config: WagentConfig } | { handled: false }> {
  if (config.agent.pendingSystemPrompt && isConfirmation(text)) {
    config.agent.systemPrompt = config.agent.pendingSystemPrompt;
    config.agent.pendingSystemPrompt = undefined;
    config.agent.memorySummary = config.agent.bootstrapNotes.join("\n").slice(-6000);
    config.agent.mode = "configured";
    saveConfig(config);
    logger.info("Bootstrap done — system prompt locked");
    await sendTracked(adapter, config, chatId, "Confirmed. Locked in my system prompt.");
    return { handled: true, config };
  }

  if (config.agent.pendingSystemPrompt && !isConfirmation(text)) {
    config.agent.bootstrapNotes.push(`Feedback: ${text}`);
    config.agent.pendingSystemPrompt = undefined;
    saveConfig(config);
    await sendTracked(adapter, config, chatId, "Draft discarded. Say 'draft prompt' for a new one.");
    return { handled: true, config };
  }

  const say = maybeExtractSayCommand(text);
  if (say) {
    await sendTracked(adapter, config, chatId, say);
    return { handled: true, config };
  }

  if (isPromptDraftRequest(text)) {
    logger.info("Drafting system prompt...");
    const draft = await generateWithFallback(config, {
      system: "Draft a concise first-person system prompt for a WhatsApp agent from the user's setup notes. Include identity, purpose, vibe, read/reply boundaries, and tool behavior. Return only the prompt.",
      messages: [{ role: "user", content: config.agent.bootstrapNotes.join("\n\n") }],
      temperature: 0.3,
    });
    config.agent.pendingSystemPrompt = draft.text.trim();
    saveConfig(config);
    await sendTracked(adapter, config, chatId,
      `System prompt draft:\n\n${config.agent.pendingSystemPrompt}\n\nReply "confirm" if correct, or tell me what to change.`);
    return { handled: true, config };
  }

  config.agent.bootstrapNotes.push(text);
  saveConfig(config);
  return { handled: false };
}

async function processMessage(
  adapter: BaileysAdapter,
  config: WagentConfig,
  instId: string,
  event: NormalizedMessageEvent,
  text: string,
  isSelf: boolean,
  senderName: string | null,
  chatName: string | null,
  tools: ReturnType<typeof buildTools>,
) {
  const { chatId } = event;
  const chatKey = `${instId}:${chatId}`;
  const history = chatHistories.get(chatKey) ?? [];

  const roleInfo = isSelf
    ? `[COMMAND from ${senderName ?? "you"} (self-message) | chat: ${chatId}]`
    : `[MESSAGE from ${senderName ?? event.message.sender} in ${chatName ?? chatId} | chat: ${chatId}]`;

  const userMsg = { role: "user" as const, content: `${roleInfo}\n\n${text}` };
  const messages = [
    ...history.slice(-MAX_HISTORY),
    userMsg,
  ];

  try {
    logger.info("Processing message...");

    const result = await generateWithFallback(config, {
      system: getSystemPrompt(config),
      messages,
      tools,
      temperature: 0.7,
      experimental_context: { currentChatId: chatId, config },
    });

    const responseText = result.text?.trim();
    const allToolCalls = result.steps?.flatMap((s) => s.toolCalls ?? []) ?? [];

    if (allToolCalls.length > 0 || responseText) {
      history.push(userMsg);
      history.push({ role: "assistant", content: responseText || "(done)" });
      chatHistories.set(chatKey, history);
    }

    if (responseText) {
      await adapter.sendPresence(chatId, "composing");
      await delayBeforeReply(config);
      const content: MessageContent = { type: "text", text: withAgentPrefix(responseText) };
      const res = await adapter.sendMessage(chatId, content);
      rememberAgentMessage(res.messageId);
      await adapter.sendPresence(chatId, "paused");
      logger.info(`Response: ${responseText.substring(0, 120)}`);
    } else if (allToolCalls.length > 0) {
      const toolNames = allToolCalls.map((tc) => tc.toolName).join(", ");
      logger.info(`Tools: ${toolNames}`);
      const res = await adapter.sendMessage(chatId, { type: "text", text: withAgentPrefix("Done.") });
      rememberAgentMessage(res.messageId);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.error(`Error: ${msg}`);
    try {
      const fallback = withAgentPrefix("Sorry, I hit an error processing that. Please try again.");
      const res = await adapter.sendMessage(chatId, { type: "text", text: fallback });
      rememberAgentMessage(res.messageId);
    } catch (_) {}
  }
}

async function main() {
  const args = process.argv.slice(2);
  if (args.includes("--help") || args.includes("-h")) {
    console.log(`wagent\n\nUsage:\n  wagent              Start agent (pairs WhatsApp first if needed)\n  wagent start        Start agent\n  wagent setup        Re-run AI provider setup after WhatsApp pairing\n  wagent config       Show config path and provider\n  wagent --verbose    Show lifecycle logs\n  wagent --debug      Show debug logs\n`);
    return;
  }

  let config = loadConfig();

  if (args[0] === "config") {
    const { getConfigPath } = await import("./config.js");
    console.log(JSON.stringify({ path: getConfigPath(), provider: config.provider, model: config.model, configured: Boolean(config.apiKey), agentMode: config.agent.mode }, null, 2));
    return;
  }

  if (args[0] === "setup") {
    config.apiKey = "";
    saveConfig(config);
  }

  console.log("wagent starting...");

  const instanceManager = new InstanceManager();

  let instances = instanceManager.getAllInstances();
  let instanceId: string;

  if (instances.length === 0) {
    const inst = await instanceManager.createInstance("wagent", "baileys");
    instanceId = inst.id;
  } else {
    instanceId = instances[0].id;
  }

  let myJid: string | null = null;
  let myUserJid: string | null = null;

  instanceManager.onAnyEvent((event, instId, payload) => {
    if (event === "connection.changed" && instId === instanceId) {
      const conn = payload as NormalizedConnectionEvent;
      if (conn.qrCode) {
        console.log("\nScan this QR in WhatsApp > Linked Devices:");
        qrcode.generate(conn.qrCode, { small: true });
      }
      if (conn.status === "open") {
        const adapter = instanceManager.getAdapter(instId) as BaileysAdapter;
        myJid = getOwnJid(adapter);
        myUserJid = toUserJid(myJid);
        console.log(`\nWhatsApp connected as ${myJid}`);
      }
    }
  });

  await instanceManager.connectInstance(instanceId);

  const adapter = instanceManager.getAdapter(instanceId) as BaileysAdapter;

  for (let i = 0; i < 60; i++) {
    myJid = getOwnJid(adapter);
    myUserJid = toUserJid(myJid);
    if (myJid && myUserJid) break;
    await new Promise((r) => setTimeout(r, 1000));
  }

  if (!myJid) {
    console.error("WhatsApp did not connect. Check QR/session and try again.");
    process.exit(1);
  }

  config = await ensureAiConfig(config);

  logger.info(`Provider: ${config.provider}, Model: ${config.model}, Mode: ${config.agent.mode}`);

  const tools = buildTools(adapter);
  config = await startBootstrap(adapter, config, myUserJid ?? myJid);

  logger.info("Agent ready");

  instanceManager.onAnyEvent(async (event, instId, payload) => {
    if (event === "presence.updated" && instId === instanceId) {
      const presence = payload as NormalizedPresenceEvent;
      if (presence.status === "composing" || presence.status === "recording") {
        typingUntil.set(presence.chatId, Date.now() + 30_000);
      } else if (presence.status === "paused" || presence.status === "unavailable") {
        typingUntil.delete(presence.chatId);
      }
    }

    if (event === "message.received" && instId === instanceId) {
      const msg = payload as NormalizedMessageEvent;
      const text = msg.message.content;
      if (!text) return;

      if (agentSentMessages.delete(msg.message.id)) return;

      const dedupKey = `${msg.chatId}:${msg.message.id}:${msg.message.timestamp}`;
      if (processedMessages.has(dedupKey)) return;
      processedMessages.add(dedupKey);
      if (processedMessages.size > 1000) {
        const toDelete = [...processedMessages].slice(0, 500);
        for (const k of toDelete) processedMessages.delete(k);
      }

      const isOwnerSelfChat = Boolean(msg.message.isFromMe && myUserJid && msg.chatId === myUserJid);
      if (!messageCanReachAgent(config, msg, isOwnerSelfChat, text)) {
        logger.info("Message filtered out");
        return;
      }

      await waitForTypingToStop(msg.chatId);

      if (config.agent.mode === "bootstrap" && isOwnerSelfChat) {
        const result = await handleBootstrapCommand(adapter, config, msg.chatId, text);
        if (result.handled) { config = result.config; return; }
      }

      const senderName = await getContactName(adapter, msg.message.sender);
      const chatName =
        msg.chatId.endsWith("@g.us")
          ? await getGroupName(adapter, msg.chatId)
          : senderName;

      processMessage(adapter, config, instanceId, msg, text, isOwnerSelfChat, senderName, chatName, tools)
        .catch((err) => logger.error(`processMessage error: ${err instanceof Error ? err.message : err}`));
    }
  });
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
