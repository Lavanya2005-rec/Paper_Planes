export const analyzeTransaction = (transaction: any) => {
    // Placeholder logic
    const risk = transaction.amount > 1000 ? 'high' : 'low'
    const confidence = risk === 'high' ? 0.95 : 0.85
  
    return {
      fraudRisk: risk,
      confidence
    }
  }
  