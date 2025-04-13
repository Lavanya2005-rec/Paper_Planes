import { Hono } from 'hono'
import { checkFraud } from '../controllers/fraudController'

const fraud = new Hono()

fraud.post('/detect', async c => {
  const tx = await c.req.json()
  const result = await checkFraud(tx)
  return c.json(result)
})

export default fraud
