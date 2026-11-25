import {
  App,
  applyDocumentTheme,
  applyHostStyleVariables,
  type McpUiHostContext,
} from '@modelcontextprotocol/ext-apps';

export interface StudioBridge {
  mode: 'mcp-app' | 'browser';
  callTool<T = Record<string, unknown>>(name: string, args?: Record<string, unknown>): Promise<T>;
  resolveMediaUrl(localPath: string): Promise<string>;
  openExternalUrl(url: string): Promise<void>;
  onStatePush(handler: (payload: Record<string, unknown>) => void): void;
}

function isEmbeddedInMcpHost(): boolean {
  try {
    return window.self !== window.top;
  } catch {
    return true;
  }
}

export async function createStudioBridge(): Promise<StudioBridge> {
  if (!isEmbeddedInMcpHost()) {
    let pushHandler: ((payload: Record<string, unknown>) => void) | null = null;
    return {
      mode: 'browser',
      async callTool<T = Record<string, unknown>>(name: string, args: Record<string, unknown> = {}): Promise<T> {
        const res = await fetch(`/api/tool/${encodeURIComponent(name)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(args),
        });
        const body = (await res.json()) as { structuredContent?: T; error?: string };
        if (!res.ok || body.error) {
          throw new Error(body.error || `Tool '${name}' failed (${res.status})`);
        }
        return (body.structuredContent ?? body) as T;
      },
      async resolveMediaUrl(localPath: string): Promise<string> {
        return `/api/media?path=${encodeURIComponent(localPath)}`;
      },
      async openExternalUrl(url: string): Promise<void> {
        window.open(url, '_blank', 'noopener,noreferrer');
      },
      onStatePush(handler) {
        pushHandler = handler;
        void pushHandler;
      },
    };
  }

  const app = new App({ name: 'Nairobiwood Director Studio', version: '0.1.1' }, {});
  let pushHandler: ((payload: Record<string, unknown>) => void) | null = null;
  let queuedPayload: Record<string, unknown> | null = null;

  const applyContext = (ctx?: McpUiHostContext) => {
    if (!ctx) return;
    if (ctx.theme) {
      applyDocumentTheme(ctx.theme);
      document.documentElement.style.colorScheme = ctx.theme;
    }
    if (ctx.styles?.variables) {
      applyHostStyleVariables(ctx.styles.variables);
    }
  };

  // Register handlers BEFORE app.connect() so initial tool notifications are never missed
  app.onhostcontextchanged = applyContext;
  app.ontoolresult = (result) => {
    if (result.structuredContent) {
      const payload = result.structuredContent as Record<string, unknown>;
      if (pushHandler) {
        pushHandler(payload);
      } else {
        queuedPayload = payload;
      }
    }
  };

  await app.connect();
  applyContext(app.getHostContext());

  return {
    mode: 'mcp-app',
    async callTool<T = Record<string, unknown>>(name: string, args: Record<string, unknown> = {}): Promise<T> {
      const result = await app.callServerTool({ name, arguments: args });
      if (result.isError) {
        const text = result.content?.map((c) => ('text' in c ? c.text : '')).join(' ') || 'Tool error';
        throw new Error(text);
      }
      return (result.structuredContent ?? {}) as T;
    },
    async resolveMediaUrl(localPath: string): Promise<string> {
      const res = await app.callServerTool({
        name: 'app_get_media_data_url',
        arguments: { file_path: localPath },
      });
      const payload = (res.structuredContent ?? {}) as { data_url?: string };
      return payload.data_url ?? '';
    },
    async openExternalUrl(url: string): Promise<void> {
      try {
        await app.openLink({ url });
      } catch {
        window.open(url, '_blank', 'noopener,noreferrer');
      }
    },
    onStatePush(handler) {
      pushHandler = handler;
      if (queuedPayload) {
        const pending = queuedPayload;
        queuedPayload = null;
        handler(pending);
      }
    },
  };
}
