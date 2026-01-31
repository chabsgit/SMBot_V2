# Schedule Service - Render Deployment

## 🚀 Quick Deploy to Render

### 1. Update Render Configuration

**Option A: Use render_schedule.yaml**
```bash
# Rename the schedule-specific config
mv render_schedule.yaml render.yaml
```

**Option B: Manually configure in Render Dashboard**

### 2. Deploy Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add schedule service for Render deployment"
   git push origin main
   ```

2. **Create Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select "Python" as runtime
   - Use existing `render_schedule.yaml` or configure manually

### 3. Environment Variables
Set these in Render Dashboard → your service → Environment:
```bash
HOST=0.0.0.0
PORT=10000
DEBUG=false
LOG_LEVEL=INFO
PYTHON_VERSION=3.9.16
```

## 📡 API Endpoints

**Base URL:** `https://smbot-schedule.onrender.com`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/schedule` | POST | Start schedule until 3 PM IST |
| `/schedule/status` | GET | Check if schedule is running |
| `/schedule/stop` | POST | Stop the schedule |

## 🧪 Testing on Render

### Start Schedule
```bash
curl -X POST https://smbot-schedule.onrender.com/schedule
```

**Response:**
```json
{
  "status": "success",
  "message": "Hello! Schedule started at 2026-01-31 13:54:11 IST. Will run until 3 PM IST.",
  "start_time": "2026-01-31 13:54:11 IST",
  "end_time": "15:00:00 IST"
}
```

### Check Status
```bash
curl https://smbot-schedule.onrender.com/schedule/status
```

### Stop Schedule
```bash
curl -X POST https://smbot-schedule.onrender.com/schedule/stop
```

### Health Check
```bash
curl https://smbot-schedule.onrender.com/health
```

## 📊 Features

- **Automatic Logging**: Logs current IST time every minute
- **Auto-stop**: Stops automatically at 3 PM IST
- **Manual Control**: Start/stop via API calls
- **Health Monitoring**: Built-in health checks
- **Production Ready**: Uses Gunicorn server

## 🔧 Configuration Notes

- **Timezone**: Uses Asia/Kolkata (IST) timezone
- **Logging Interval**: Every 60 seconds
- **Stop Time**: 15:00:00 IST (3 PM)
- **Server**: Gunicorn with 1 worker (suitable for background tasks)
- **Timeout**: 120 seconds for long-running operations

## 📝 Logs

View logs in Render Dashboard → smbot-schedule → Logs

The service will log:
- Schedule start/stop events
- Current timestamp every minute
- Health check status
- Any errors or exceptions

## 🚨 Important

- The schedule runs in a background thread
- Only one schedule can run at a time
- Manual stop overrides the 3 PM auto-stop
- Service restarts will clear running schedules
