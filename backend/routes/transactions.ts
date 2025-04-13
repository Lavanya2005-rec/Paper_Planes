import { Hono } from 'hono';
import Transaction from '../models/Transaction';

const tx = new Hono();

// ✅ Dashboard Summary: GET /transactions/summary
tx.get('/summary', async (c) => {
  try {
    const total = await Transaction.aggregate([
      {
        $group: {
          _id: '$riskLevel',
          count: { $sum: 1 },
        },
      },
    ]);

    const counts: Record<string, number> = {
      safe: 0,
      suspicious: 0,
      fraud: 0,
    };

    total.forEach((item) => {
      const key = item._id?.toLowerCase();
      if (counts[key] !== undefined) counts[key] = item.count;
    });

    const amountProtected = await Transaction.aggregate([
      { $match: { riskLevel: 'safe' } },
      { $group: { _id: null, total: { $sum: '$amount' } } },
    ]);

    const monthlyLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const monthly = {
      labels: monthlyLabels,
      safe: Array(12).fill(0),
      suspicious: Array(12).fill(0),
      fraud: Array(12).fill(0),
    };

    const monthlyAgg = await Transaction.aggregate([
      {
        $addFields: {
          safeDate: {
            $convert: {
              input: "$date",
              to: "date",
              onError: null,
              onNull: null
            }
          }
        }
      },
      {
        $match: {
          safeDate: { $ne: null }
        }
      },
      {
        $project: {
          month: { $month: "$safeDate" },
          riskLevel: 1
        }
      },
      {
        $group: {
          _id: { month: "$month", riskLevel: "$riskLevel" },
          total: { $sum: 1 }
        }
      }
    ]);
    
    
    

    monthlyAgg.forEach((item) => {
      const idx = item._id.month - 1;
      const type = item._id.riskLevel?.toLowerCase();
      if (type && monthly[type] && monthly[type][idx] !== undefined) {
        monthly[type][idx] = item.total;
      }
    });

    const fraudTypesAgg = await Transaction.aggregate([
      { $match: { riskLevel: 'fraud' } },
      {
        $group: {
          _id: { $ifNull: ['$fraudType', 'Unknown'] },
          total: { $sum: 1 },
        },
      },
    ]);
    

    const fraudTypes: Record<string, number> = {};
    fraudTypesAgg.forEach((item) => {
      fraudTypes[item._id || 'Other'] = item.total;
    });

    return c.json({
      safe: counts.safe,
      suspicious: counts.suspicious,
      fraud: counts.fraud,
      protectedAmount: amountProtected[0]?.total ?? 0,
      monthly,
      fraudTypes,
    });
  } catch (err) {
    console.error('Dashboard summary error:', err);
    return c.json({ error: 'Failed to fetch dashboard data' }, 500);
  }
});


// ✅ Live Transactions: GET /transactions/live
tx.get('/live', async (c) => {
  const type = c.req.query('type') || 'all';
  const range = c.req.query('range') || 'today';
  const from = c.req.query('from');
  const to = c.req.query('to');

  const match: Record<string, unknown> = {};

  if (type !== 'all') {
    match.riskLevel = type;
  }

  const now = new Date();
  let start: Date | null = null;
  let end: Date = new Date();

  switch (range) {
    case 'today':
      start = new Date();
      start.setUTCHours(0, 0, 0, 0);
      end = new Date();
      end.setUTCHours(23, 59, 59, 999);
      break;
    case 'week':
      start = new Date();
      start.setUTCDate(now.getUTCDate() - 7);
      break;
    case 'month':
      start = new Date();
      start.setUTCMonth(now.getUTCMonth() - 1);
      break;
      case 'quarter': {
        const quarter = Math.floor(now.getMonth() / 3);
        start = new Date(now.getFullYear(), quarter * 3, 1);
        end = now;
        break;
      }
    case 'all':
      // do not apply any date filter
      start = null;
      break;
    default:
      console.warn('[INVALID RANGE]', range);
      start = null;
      break;
  }

  if (start) {
    match.date = { $gte: start, $lte: end };
  }

  console.log('[LIVE FILTER]', match);

  try {
    const transactions = await Transaction.find(match).sort({ date: -1 });
    console.log('[LIVE RESULT]', transactions.length);
    return c.json(transactions);
  } catch (err) {
    console.error('Failed to fetch transactions:', err);
    return c.json({ error: 'Internal server error' }, 500);
  }
});


// GET fraud transactions only
tx.get('/fraud', async (c) => {
  try {
    const data = await Transaction.find({ riskLevel: 'fraud' });
    return c.json(data);
  } catch (err) {
    console.error('Error fetching fraud alerts:', err);
    return c.json({ error: 'Server error' }, 500);
  }
});

tx.patch('/:id/status', async (c) => {
  const id = c.req.param('id');
  const { status } = await c.req.json();

  try {
    const updated = await Transaction.findByIdAndUpdate(id, { status }, { new: true });
    if (!updated) {
      return c.json({ error: 'Transaction not found' }, 404);
    }
    console.log('[PATCH] Updated transaction status:', id, status);
    return c.json(updated);
  } catch (err) {
    console.error('Failed to update transaction status:', err);
    return c.json({ error: 'Failed to update' }, 500);
  }
});

export default tx;
