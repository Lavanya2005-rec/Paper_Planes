import { Hono } from 'hono';
import { getUserProfile, updateUserProfile } from '../controllers/userController';
import { authenticate } from '../middleware/authMiddleware';

type Variables = {
  user: { id: string };
};

const user = new Hono<{ Variables: Variables }>();

user.get('/profile', authenticate, getUserProfile);
user.put('/profile', authenticate, updateUserProfile);

export default user;
