import "dotenv/config";
import { InstanceManager } from "./services/instance-manager.js";
import { BaileysAdapter } from "./channels/baileys/baileys.adapter.js";
import { createChildLogger } from "./utils/logger.js";
import { generateText, tool, stepCountIs } from "ai";
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
const chatModel = groq.chatModel(process.env.GROQ_MODEL ?? "gpt-oss-120b");

const chatHistories = new Map<string, Array<{ role: "user" | "assistant"; content: string }>>();
const MAX_HISTORY = 30;
const processedMessages = new Set<string>();

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
        const target = to ?? ctx.currentChatId;
        if (!target || !target.includes("@")) throw new Error(`Invalid target JID: ${target}`);
        const content: MessageContent = { type: "text", text };
        if (quotedMessageId) content.quotedMessageId = quotedMessageId;
        const res = await adapter.sendMessage(target, content);
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
        const res = await adapter.sendMessage(to ?? ctx.currentChatId, { type: "image", image: imageUrl, caption });
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
        const res = await adapter.sendMessage(to ?? ctx.currentChatId, { type: "video", video: videoUrl, caption });
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
        const res = await adapter.sendMessage(to ?? ctx.currentChatId, { type: "audio", audio: audioUrl, ptt: asVoiceNote });
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
        const res = await adapter.sendMessage(to ?? ctx.currentChatId, { type: "document", document: documentUrl, fileName, mimeType });
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
        const res = await adapter.sendMessage(to ?? ctx.currentChatId, { type: "location", latitude, longitude, name, address });
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
        const res = await adapter.sendMessage(to ?? ctx.currentChatId, { type: "contact", contactName, contactPhone });
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
        const res = await adapter.sendMessage(to ?? ctx.currentChatId, { type: "poll", question, options, multiSelect });
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
- To message someone by NAME, ALWAYS call searchContact first to find their JID, then use that JID in the send tool. Never use a name as the 'to' parameter.
- You have multiple steps available. Use searchContact first, then sendText with the correct JID.

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
    { role: "system" as const, content: SYSTEM_PROMPT },
    ...history.slice(-MAX_HISTORY),
    userMsg,
  ];

  try {
    logger.info({ chatId, isSelf, text: text.substring(0, 80) }, "Processing with AI");

    const result = await generateText({
      model: chatModel,
      messages,
      tools,
      temperature: 0.7,
      stopWhen: stepCountIs(5),
      allowSystemInMessages: true,
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
      await adapter.sendMessage(chatId, content);
      await adapter.sendPresence(chatId, "paused");
      logger.info({ chatId, response: responseText.substring(0, 80) }, "Response sent");
    } else if (allToolCalls.length > 0) {
      logger.info({ chatId, toolCount: allToolCalls.length }, "Tools executed silently");
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.error({ err, chatId }, "Agent error");
    try {
      const fallback = "Sorry, I hit an error processing that. Please try again.";
      await adapter.sendMessage(chatId, { type: "text", text: fallback });
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
