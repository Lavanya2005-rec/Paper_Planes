import mongoose from 'mongoose';

const transactionSchema = new mongoose.Schema({
  userId: String,
  name: String,
  username: String,
  email: String,
  amount: Number,
  location: String,
  riskLevel: {
    type: String,
    enum: ['safe', 'suspicious', 'fraud'],
  },
  riskScore: Number,
  date: Date,
  status: {
    type: String,
    enum: ['pending', 'resolved'],
    default: 'pending',
  }
});

const Transaction = mongoose.model('Transaction', transactionSchema);

export default Transaction;
