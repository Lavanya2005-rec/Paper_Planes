import { Request } from 'express';

export interface CustomRequest extends Request {
  body: {
    fullName: string;
    email: string;
    phone: string;
  };
  user: {
    id: string;
    email?: string;
    accountType?: string;
  };
}
