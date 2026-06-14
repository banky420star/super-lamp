# Chain Gambler Dashboard - Improved Documentation

## Overview

The Chain Gambler Dashboard is a React-based monitoring and control interface for an autonomous AI trading system. It provides real-time visibility into trading activity, model training, risk controls, and system health.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev -- --port 4180

# Build for production
npm run build
```

## Architecture

```
Dashboard
├── App.tsx (Main container)
├── components/
│   ├── TradesPanel.tsx          # Trading activity and history
│   ├── DashboardPanel.tsx        # System overview and KPIs
│   ├── TrainingPanel.tsx         # Model training controls
│   ├── ModelBrainsPanel.tsx      # AI model diagnostics
│   ├── RegistryPanel.tsx         # Champion/canary models
│   ├── SafetyPanel.tsx           # Risk controls
│   └── HelpTooltip.tsx           # Contextual help component
├── services/
│   └── api.ts                    # API client and types
└── types.ts                      # TypeScript definitions
```

## Key Concepts Explained

### The Three AI Brains

The trading system uses three complementary AI models:

1. **LSTM (Long Short-Term Memory)**
   - Analyzes sequential price data
   - Detects patterns and trends
   - Outputs: HOLD, BUY, or SELL signals

2. **PPO (Proximal Policy Optimization)**
   - Reinforcement learning agent
   - Learns optimal trading policies
   - Continuously improves from experience

3. **Dreamer**
   - World model for scenario planning
   - Predicts future market states
   - Currently stubbed/disabled in demo

### Champion/Canary System

Think of this as A/B testing for AI models:

- **Champion**: Current best model making live trades
- **Canary**: New model being tested in "shadow mode"
- **Candidate**: Models awaiting evaluation

Promotion flow:
```
Training → Evaluation → Canary (shadow) → Promotion → Champion
```

### Signal Lanes

Each trading symbol has a "lane" showing:
- **Regime**: Current market condition (HOLD/BUY/SELL)
- **Confidence**: AI certainty level (0-1)
- **Final Target**: Blended signal from all models (-1 to +1)

Trades only execute when confidence exceeds threshold (~0.6).

### Risk Controls

Multiple safety layers protect against losses:

**Hard Limits** (Cannot override):
- Max daily loss: $1,000
- Max drawdown: 8%
- Max open positions: 8

**Soft Checks** (Per trade):
- Spread < 25 bps
- Cooldown: 45 seconds between trades
- Per-symbol exposure < 35%

## UI Components

### HelpTooltip

Provides contextual explanations throughout the UI:

```tsx
import HelpTooltip from './components/HelpTooltip'

// Basic usage
<HelpTooltip text="Explanation of this metric" />

// With title
<HelpTooltip
  title="Section Title"
  text="Detailed explanation of what this section does"
/>
```

### Color Coding

| Color | Meaning | Use Case |
|-------|---------|----------|
| Green (#39d98a) | Success/Active/Win | Profitable trades, running systems |
| Red (#ff7b8f) | Error/Halt/Loss | Risk halts, losing trades |
| Amber (#f3bb4a) | Warning/Idle | Pending states, caution |
| Cyan (#5ad7ff) | Info/Neutral | General information |

### Status Indicators

- **LIVE** - System active and trading
- **HALTED** - Trading paused (check risk tab)
- **LOCKED** - Real money disabled (safety)
- **WS/POLL** - WebSocket connected or polling

## Dashboard Tabs

| Tab | Purpose | Key Metrics |
|-----|---------|-------------|
| **Trades** | Trading activity | PnL, Win Rate, Equity Curve |
| **Model Brains** | AI diagnostics | Model status, confidence |
| **Pipeline** | Training queue | Symbols, progress, stages |
| **Training** | Model development | LSTM/PPO/Dreamer progress |
| **Registry** | Model library | Champion/canary status |
| **Signal Lanes** | Per-symbol decisions | Regime, confidence, exposure |
| **Safety** | Risk controls | Halt status, limits |
| **System Truth** | Health overview | All component statuses |

## Data Flow

```
┌─────────────┐    WebSocket    ┌──────────────┐
│ API Server  │ ◄──────────────►│  React App   │
│  (port 5051)│    (real-time)  │  (port 4180) │
└─────────────┘                 └──────────────┘
       │                                │
       │    HTTP REST (fallback)        │
       └───────────────────────────────┘
              (10s polling)
```

## API Integration

### Real-time Updates

The dashboard uses WebSocket for real-time updates with HTTP polling fallback:

```typescript
// services/api.ts
createStatusWS(
  onMessage: (data: StatusPayload) => void,
  onConnectionChange: (connected: boolean) => void
): () => void  // Returns cleanup function
```

### Key Endpoints

```typescript
// System status
fetchStatus(): Promise<StatusPayload>

// Trading data
fetchTrades(params): Promise<TradesResponse>
fetchTradesSummary(): Promise<TradeSummary>
fetchEquityCurve(window): Promise<EquityCurveResponse>

// Training
fetchTrainingStatus(): Promise<StatusPayload>
fetchTrainingLanes(): Promise<TrainingLanesResponse>

// Controls
controlAction(action: string): Promise<ControlResponse>
```

## Customization

### Adding New Tabs

1. Add tab definition to `TABS` array:
```typescript
const TABS: TabInfo[] = [
  // ... existing tabs
  { id: 'my_tab', label: 'My Tab', description: 'What this tab does' },
]
```

2. Add case to `renderContent()`:
```typescript
case 'my_tab': return <MyTabPanel />
```

3. Create component in `components/MyTabPanel.tsx`

### Adding Help Text

Add tooltips to explain complex concepts:

```tsx
const HELP_TEXTS = {
  myMetric: "Explanation of what this metric means",
}

// In component
<div>
  My Metric
  <HelpTooltip text={HELP_TEXTS.myMetric} />
</div>
```

### Styling

The dashboard uses CSS custom properties for theming:

```css
:root {
  --bg: #04080f;           /* Background */
  --text: #e8f4ff;         /* Primary text */
  --muted: #7a94b0;        /* Secondary text */
  --cyan: #00f0ff;         /* Accent */
  --green: #00ff88;        /* Success */
  --red: #ff3366;          /* Error */
  --amber: #ffd700;        /* Warning */
}
```

## Troubleshooting

### Dashboard not loading
- Check API server is running on port 5051
- Verify `npm install` was run
- Check browser console for errors

### No data showing
- Verify WebSocket connection (shows "WS" or "POLL")
- Check `/api/status` returns valid JSON
- Ensure MT5 is connected (for live data)

### Stale data
- WebSocket may have disconnected
- Dashboard automatically falls back to polling
- Check network tab for failed requests

### Performance issues
- Reduce polling interval in `App.tsx`
- Disable expensive visualizations
- Use production build (`npm run build`)

## Development Tips

### Hot Reload
The Vite dev server provides instant updates on file changes.

### Type Safety
All API responses are typed in `types.ts`. Use these for autocompletion:

```typescript
import { StatusPayload, Trade, TrainingVisual } from './types'
```

### Testing API
Test endpoints directly in browser:
```
http://localhost:5051/api/status
http://localhost:5051/api/trades?limit=10
```

## File Structure Best Practices

```
components/
├── ComponentName.tsx          # Main component file
├── ComponentName.test.tsx     # Unit tests (if needed)
└── index.ts                   # Barrel export (optional)

services/
├── api.ts                     # API functions
└── types.ts                   # Shared types

styles/
├── globals.css                # Global styles
└── theme.ts                   # Theme constants
```

## Common Patterns

### Loading States
```tsx
const [loading, setLoading] = useState(true)

useEffect(() => {
  fetchData().finally(() => setLoading(false))
}, [])

if (loading) return <LoadingBar />
```

### Error Handling
```tsx
try {
  const data = await fetchData()
  setState(data)
} catch (err) {
  console.error('Failed to load:', err)
  setState(null)
}
```

### Polling with Cleanup
```tsx
useEffect(() => {
  const load = () => fetchData().then(setState)
  load()
  const id = setInterval(load, 10_000)
  return () => clearInterval(id)
}, [])
```

## Security Notes

- Never hardcode credentials in frontend code
- API should handle authentication, not the UI
- Real money lock is enforced server-side
- All trading decisions happen on the server

## Resources

- [React Docs](https://react.dev)
- [Vite Guide](https://vitejs.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs)

---

*Chain Gambler Dashboard v1.0*
