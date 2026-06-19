# wagent — WhatsApp AI Agent

A programmable WhatsApp agent powered by Groq LLM, Vercel AI SDK, and Baileys (WhatsApp Web). Send commands via self-messages — your own number becomes the control channel.

## Features

- **Self-programmable** — message yourself to command the agent
- **Full WhatsApp API** — send/receive text, images, videos, audio, documents, locations, contacts, polls
- **Group management** — create, join, leave, manage groups (add/remove/promote/demote members)
- **Chat management** — archive, pin, mute, mark read, send presence indicators
- **Profile management** — update name, status, get profile info
- **Status/stories** — post text and image status updates
- **Message management** — forward, edit, delete, pin, react to messages
- **Contact management** — search contacts, check numbers on WhatsApp, block/unblock
- **Conversation memory** — remembers recent chat history per conversation
- **Auto-reply** — responds to incoming messages from others

## Prerequisites

- **Node.js >= 22** (tested on v24)
- **pnpm** (package manager)
- **Groq API key** — get one at [console.groq.com](https://console.groq.com)

## Setup

```bash
# Clone
git clone <your-repo-url>
cd wagent

# Install dependencies
pnpm install

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

## Usage

```bash
pnpm wagent
```

First run will display a QR code — scan it with WhatsApp (Linked Devices) to authenticate. The session is persisted to SQLite, so you won't need to re-scan on restart.

### How to use

Once connected:
- **Message yourself** to send commands to the agent (e.g., "send hi to mom", "create a group called family with ...")
- **Others can message you** and the agent will auto-reply

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Your Groq API key |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Optional. API base URL override |
| `WA_LOG_LEVEL` | `info` | Log level (debug, info, warn, error) |

## Tech Stack

- **Runtime:** Node.js + tsx (TypeScript execution)
- **WhatsApp:** Baileys v7 (WebSocket-based, no browser needed)
- **AI:** Vercel AI SDK + Groq (llama-3.3-70b-versatile)
- **Database:** SQLite (via better-sqlite3 + Drizzle ORM)
- **Schema:** Zod
- **Logging:** Pino

## Project Structure

```
src/
├── index.ts                     # Agent entry point
├── channels/
│   └── baileys/                 # WhatsApp adapter (Baileys)
├── db/                          # SQLite schema + migrations
├── services/
│   └── instance-manager.ts      # Instance lifecycle management
├── types/                       # TypeScript types
└── utils/                       # Logger, constants
```

## License

MIT
