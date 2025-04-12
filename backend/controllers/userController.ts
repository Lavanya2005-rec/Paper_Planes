import { Context } from 'hono';
import User from '../models/User';

export const getUserProfile = async (c: Context) => {
  const user = c.get('user'); // user.id comes from authMiddleware

  const userData = await User.findById(user.id).select('-password');
  if (!userData) {
    return c.json({ error: 'User not found' }, 404);
  }

  return c.json(userData);
};

export const updateUserProfile = async (c: Context) => {
  const user = c.get('user');
  const { fullName, email, phone } = await c.req.json();

  const updated = await User.findByIdAndUpdate(
    user.id,
    { fullName, email, phone },
    { new: true }
  ).select('-password');

  return c.json(updated);
};
