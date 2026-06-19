import "dotenv/config";
import { InstanceManager } from "./services/instance-manager.js";
import { BaileysAdapter } from "./channels/baileys/baileys.adapter.js";
import { createChildLogger } from "./utils/logger.js";
import { generateText, tool } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { z } from "zod";
import qrcode from "qrcode-terminal";
import type {
  NormalizedMessageEvent,
  NormalizedConnectionEvent,
  MessageContent,
} from "./types/channel.types.js";
const logger = createChildLogger({ service: "wagent" });

const groq = createOpenAICompatible({
  name: "groq",
  baseURL: process.env.GROQ_BASE_URL ?? "https://api.groq.com/openai/v1",
  apiKey: process.env.GROQ_API_KEY ?? "",
});
const chatModel = groq.chatModel(process.env.GROQ_MODEL ?? "openai/gpt-oss-120b");

const chatHistories = new Map<string, Array<{ role: "user" | "assistant"; content: string }>>();
const MAX_HISTORY = 30;
const processedMessages = new Set<string>();
const agentSentMessages = new Set<string>();

function getOwnJid(adapter: BaileysAdapter): string | null {
  return adapter.getMyJid();
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

type ChatCtx = { currentChatId: string };

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
        const content: MessageContent = { type: "text", text };
        if (quotedMessageId) content.quotedMessageId = quotedMessageId;
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

async function processMessage(
  adapter: BaileysAdapter,
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
    logger.info({ chatId, isSelf, text: text.substring(0, 80) }, "Processing with AI");

    const result = await generateText({
      model: chatModel,
      system: SYSTEM_PROMPT,
      messages,
      tools,
      temperature: 0.7,
      experimental_context: { currentChatId: chatId },
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
      await new Promise((r) => setTimeout(r, 300));
      const content: MessageContent = { type: "text", text: responseText };
      const res = await adapter.sendMessage(chatId, content);
      rememberAgentMessage(res.messageId);
      await adapter.sendPresence(chatId, "paused");
      logger.info({ chatId, response: responseText.substring(0, 80) }, "Response sent");
    } else if (allToolCalls.length > 0) {
      logger.info({ chatId, toolCount: allToolCalls.length }, "Tools executed silently");
      const res = await adapter.sendMessage(chatId, { type: "text", text: "Done." });
      rememberAgentMessage(res.messageId);
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.error({ err, chatId }, "Agent error");
    try {
      const fallback = "Sorry, I hit an error processing that. Please try again.";
      const res = await adapter.sendMessage(chatId, { type: "text", text: fallback });
      rememberAgentMessage(res.messageId);
    } catch (_) {}
  }
}

async function main() {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    console.error("ERROR: GROQ_API_KEY environment variable is required");
    console.error("Get one at https://console.groq.com");
    process.exit(1);
  }

  console.log("wagent — WhatsApp AI Agent starting...\n");

  const instanceManager = new InstanceManager();

  let instances = instanceManager.getAllInstances();
  let instanceId: string;

  if (instances.length === 0) {
    const inst = await instanceManager.createInstance("wagent", "baileys");
    instanceId = inst.id;
    console.log(`Created new instance: ${instanceId}`);
  } else {
    instanceId = instances[0].id;
    console.log(`Using existing instance: ${instanceId}`);
  }

  let myJid: string | null = null;

  instanceManager.onAnyEvent((event, instId, payload) => {
    if (event === "connection.changed" && instId === instanceId) {
      const conn = payload as NormalizedConnectionEvent;
      if (conn.qrCode) {
        console.log("\nScan QR code to log in:");
        qrcode.generate(conn.qrCode, { small: true });
      }
      if (conn.status === "open") {
        const adapter = instanceManager.getAdapter(instId) as BaileysAdapter;
        myJid = getOwnJid(adapter);
        console.log(`\nConnected as: ${myJid}`);
        console.log("Ready. Message yourself to command the agent.\n");
      }
    }
  });

  await instanceManager.connectInstance(instanceId);

  const adapter = instanceManager.getAdapter(instanceId) as BaileysAdapter;

  for (let i = 0; i < 60; i++) {
    myJid = getOwnJid(adapter);
    if (myJid) break;
    await new Promise((r) => setTimeout(r, 1000));
  }

  if (!myJid) {
    console.error("Failed to connect. Check QR and try again.");
    process.exit(1);
  }

  const tools = buildTools(adapter);

  console.log(`Logged in: ${myJid}`);
  console.log("Waiting for messages...\n");

  instanceManager.onAnyEvent(async (event, instId, payload) => {
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

      const isSelf = msg.message.sender === myJid || msg.message.isFromMe;
      if (isSelf) {
        console.log(`Self-command: ${text.substring(0, 100)}`);
      } else {
        console.log(`From ${msg.message.sender}: ${text.substring(0, 100)}`);
      }

      const senderName = await getContactName(adapter, msg.message.sender);
      const chatName =
        msg.chatId.endsWith("@g.us")
          ? await getGroupName(adapter, msg.chatId)
          : senderName;

      processMessage(adapter, instanceId, msg, text, isSelf, senderName, chatName, tools)
        .catch((err) => logger.error({ err }, "processMessage error"));
    }
  });

  console.log("Agent listening.\n");
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
