# wagent

Programmable WhatsApp AI agent that runs from your terminal.

It pairs with WhatsApp first, then asks for an AI provider/API key, then bootstraps itself by messaging you for identity, purpose, vibe, and permissions.

## Install / Run

```bash
npx wagent
```

Or install globally:

```bash
npm i -g wagent
wagent
```

From source:

```bash
pnpm install
pnpm build
pnpm wagent
```

## First Run Flow

1. Pair WhatsApp by scanning the QR code.
2. Choose an AI provider.
3. Enter API key.
4. The agent messages you to configure its identity and permissions.
5. It drafts a system prompt and asks for confirmation.
6. After confirmation, it compacts setup context and follows the saved prompt.

## Providers

Setup wizard options:

1. Gemini + Web Search (recommended)
2. Groq
3. Other OpenAI-compatible endpoint

Config is stored locally, not in `.env`:

- Linux/macOS: `~/.config/wagent/config.json`
- Windows: `%APPDATA%/wagent/config.json`

The WhatsApp SQLite session is stored in the same config directory as `wagent.db`.

Environment variables still work for advanced use:

```bash
GROQ_API_KEY=...
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-120b
```

## Commands

```bash
wagent              # start agent
wagent start        # start agent
wagent setup        # re-run provider/API-key setup after WhatsApp pairing
wagent config       # show config path/provider/model
wagent --verbose    # lifecycle logs
wagent --debug      # debug logs
```

## Message Permissions

By default:

- The agent only reads/responds to you.
- It does not auto-reply to others.
- Group chats are ignored unless someone tags `@wagent`.
- `@wagent` bypasses filters, including blacklist.

The bootstrap conversation lets you configure who it can read, reply to, or message.

## Anti-Ban / Human-Like Behavior

- Replies are delayed randomly between 2 and 8 seconds.
- The agent waits if the chat is currently typing/recording when WhatsApp presence is available.
- CLI logs are quiet by default.

## Capabilities

- Send text, image, video, audio, documents, contacts, polls, location
- Search/resolve contacts by name, phone, or JID
- Manage profile/status
- Manage groups/chats
- Remember and compact context
- Enforce read/reply/send policy before model access

## Notes

This should run on an always-on machine/VPS. It is not suitable for Vercel serverless because Baileys needs a persistent WhatsApp WebSocket and durable session storage.

## License

MIT
