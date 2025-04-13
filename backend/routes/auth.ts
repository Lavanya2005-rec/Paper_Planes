import { Hono } from 'hono';
import { register, login } from '../controllers/authController';

const auth = new Hono();

auth.post('/register', register); // ✅ no manual arg passing
auth.post('/login', login);

export default auth;
