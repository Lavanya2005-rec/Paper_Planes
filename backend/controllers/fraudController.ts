import { analyzeTransaction } from '../services/fraudEngine'
import FraudLog from '../models/FraudLog'

export const checkFraud = async (transaction: any) => {
  const result = analyzeTransaction(transaction)

  const log = new FraudLog({
    transactionId: transaction.id || 'unknown',
    fraudRisk: result.fraudRisk,
    confidence: result.confidence
  })

  await log.save()
  return result
}
