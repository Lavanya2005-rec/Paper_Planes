import Transaction from '../models/Transaction'

export const addTransaction = async (data: any) => {
  const tx = new Transaction(data)
  await tx.save()
  return tx
}

export const getUserTransactions = async (userId: string) => {
  return await Transaction.find({ userId })
}
