// controllers/authController.ts
import { Context } from 'hono';
import User from '../models/User';

export const register = async (email: string, password: string) => {
  const existing = await User.findOne({ email });
  if (existing) throw new Error('User already exists');
  const user = new User({ email, password }); // 🔐 add hashing later
  await user.save();
  return { message: 'User registered' };
};

export const login = async (c: Context) => {
  const { email, password } = await c.req.json();
  const user = await User.findOne({ email });
  if (!user || user.password !== password) {
    return c.json({ error: 'Invalid credentials' }, 401);
  }

  // ✅ Set session cookie
  const session = await c.get('session');
  session.set('userId', user._id.toString());

  return c.json({ message: 'Login successful' });
};
