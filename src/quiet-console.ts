const originalConsole = {
  log: console.log.bind(console),
  warn: console.warn.bind(console),
  error: console.error.bind(console),
};

function shouldSuppressConsole(args: unknown[]): boolean {
  if (process.argv.includes("--debug")) return false;
  const text = args.map((arg) => (typeof arg === "string" ? arg : "")).join(" ");
  return (
    text.startsWith("Closing session:") ||
    text.includes("Decrypted message with closed session") ||
    text.includes("stream errored out") ||
    text.includes("no name present") ||
    text.includes("blocked on missing key") ||
    text.includes("failed to find key") ||
    text.includes("transaction failed") ||
    text.includes("failed to decrypt message")
  );
}

console.log = (...args: unknown[]) => {
  if (!shouldSuppressConsole(args)) originalConsole.log(...args);
};

console.warn = (...args: unknown[]) => {
  if (!shouldSuppressConsole(args)) originalConsole.warn(...args);
};

console.error = (...args: unknown[]) => {
  if (!shouldSuppressConsole(args)) originalConsole.error(...args);
};
