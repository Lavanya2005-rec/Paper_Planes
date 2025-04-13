import { MiddlewareHandler } from 'hono';

type SessionStore = {
  get: (key: string) => unknown;
  set: (key: string, value: unknown) => void;
  delete: (key: string) => void;
  all: () => Record<string, unknown>;
};

export const authenticate: MiddlewareHandler = async (c, next) => {
  const session = c.get('session') as SessionStore;
  const userId = session.get('userId') as string | undefined;

  if (!userId) {
    return c.json({ error: 'Unauthorized' }, 401);
  }

  c.set('user', { id: userId });
  await next();
};
