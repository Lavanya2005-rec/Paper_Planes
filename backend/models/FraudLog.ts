import mongoose from 'mongoose'

const FraudLogSchema = new mongoose.Schema({
  transactionId: String,
  fraudRisk: String,
  confidence: Number,
  analyzedAt: { type: Date, default: Date.now }
})

export default mongoose.model('FraudLog', FraudLogSchema)
