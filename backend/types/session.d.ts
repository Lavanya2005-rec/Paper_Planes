type SessionStore = {
    get: (key: string) => unknown;
    set: (key: string, value: unknown) => void;
    delete: (key: string) => void;
    all: () => Record<string, unknown>;
  };
  