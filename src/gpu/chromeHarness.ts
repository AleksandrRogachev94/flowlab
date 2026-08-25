/**
 * Runs an expression in a real browser, on a real GPU, from Node.
 *
 * WGSL has no unit test story: `node --test` cannot create a GPUDevice, there
 * is no breakpoint inside a compute shader, and a wrong shader looks exactly
 * like a wrong buffer binding from the outside. PLAN.md's Phase 2 notes budget
 * for that friction. This removes most of it — a shader can be diffed against
 * its CPU reference in about two seconds, with no browser open.
 *
 * Zero dependencies, like the rest of the test setup: Chrome is driven over
 * the DevTools Protocol through Node 24's global `fetch` and `WebSocket`.
 *
 * THE ONE NON-OBVIOUS PART: the page is served over http://127.0.0.1 rather
 * than loaded from about:blank or file://. WebGPU is gated on a secure
 * context; about:blank reports `isSecureContext === false` and `navigator.gpu`
 * is simply absent there, which looks identical to "this machine has no
 * WebGPU". localhost counts as secure, so a four-line HTTP server is the
 * difference between the harness working and appearing unsupported.
 */
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';

/** Override to test another Chromium build, or on Linux/Windows. */
const CHROME =
  process.env.CHROME_PATH ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const DEBUG_PORT = Number(process.env.CHROME_DEBUG_PORT ?? 9333);
const HTTP_PORT = Number(process.env.CHROME_PAGE_PORT ?? 9444);

/** How long to wait for Chrome to open a debuggable page target. */
const LAUNCH_TIMEOUT_MS = 15000;

export class NoBrowserError extends Error {}

interface Target {
  type: string;
  webSocketDebuggerUrl: string;
}

async function waitForPage(): Promise<Target> {
  const deadline = Date.now() + LAUNCH_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
      const targets = (await res.json()) as Target[];
      const page = targets.find((t) => t.type === 'page');
      if (page) return page;
    } catch {
      // Chrome is not listening yet; that is the normal case for the first
      // few hundred milliseconds, not an error worth surfacing.
    }
    await new Promise((r) => setTimeout(r, 150));
  }
  throw new NoBrowserError(`Chrome did not expose a debug target on port ${DEBUG_PORT}`);
}

/**
 * Evaluates `expression` — which should be an async IIFE — in a headless
 * Chrome page and returns its resolved value, structured-cloned back.
 *
 * Everything the expression needs must be inlined into the string: it runs in
 * a page with no modules and no access to this process. In practice that means
 * `JSON.stringify` for the shader source and for input arrays.
 */
export async function evalInBrowser(expression: string, timeoutMs = 60000): Promise<unknown> {
  const server = createServer((_req, res) => {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end('<!doctype html><meta charset="utf-8"><title>wgsl harness</title><body>');
  });
  await new Promise<void>((resolve) => server.listen(HTTP_PORT, '127.0.0.1', resolve));

  const chrome = spawn(
    CHROME,
    [
      '--headless=new',
      `--remote-debugging-port=${DEBUG_PORT}`,
      // Headless Chrome will not hand out an adapter without this.
      '--enable-unsafe-webgpu',
      '--use-angle=metal',
      '--no-first-run',
      '--no-default-browser-check',
      // A throwaway profile, or a running Chrome's profile lock blocks launch.
      `--user-data-dir=/tmp/wgsl-harness-${DEBUG_PORT}`,
      `http://127.0.0.1:${HTTP_PORT}/`,
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );
  let stderr = '';
  chrome.stderr.on('data', (d: Buffer) => (stderr += String(d)));
  chrome.on('error', () => {
    stderr += `\ncould not spawn ${CHROME} — set CHROME_PATH`;
  });

  try {
    const page = await waitForPage().catch((e: Error) => {
      throw new NoBrowserError(`${e.message}\n${stderr.slice(0, 1500)}`);
    });
    const ws = new WebSocket(page.webSocketDebuggerUrl);
    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new NoBrowserError('DevTools websocket failed'));
    });

    try {
      return await new Promise<unknown>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('evaluate timed out')), timeoutMs);
        ws.onmessage = (ev: MessageEvent) => {
          const msg = JSON.parse(String(ev.data)) as {
            id?: number;
            error?: unknown;
            result?: {
              exceptionDetails?: unknown;
              result?: { value?: unknown };
            };
          };
          if (msg.id !== 1) return;
          clearTimeout(timer);
          if (msg.error) return reject(new Error(JSON.stringify(msg.error)));
          if (msg.result?.exceptionDetails) {
            // The page's own stack, which is the only place a WebGPU
            // validation failure explains itself.
            return reject(new Error(JSON.stringify(msg.result.exceptionDetails, null, 2)));
          }
          resolve(msg.result?.result?.value);
        };
        ws.send(
          JSON.stringify({
            id: 1,
            method: 'Runtime.evaluate',
            params: { expression, awaitPromise: true, returnByValue: true },
          }),
        );
      });
    } finally {
      ws.close();
    }
  } finally {
    chrome.kill('SIGKILL');
    server.close();
  }
}
