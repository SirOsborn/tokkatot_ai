# Deployment Guide

## Raspberry Pi (24/7 Local Operation)

### Install Dependencies

```bash
ssh pi@raspberrypi.local
sudo apt update && sudo apt install -y python3.9 python3-pip
pip3 install torch torchvision opencv-python ultralytics
```

### Deploy App

```bash
# Local machine
scp -r application/ pi@raspberrypi.local:/home/pi/tokkatot_ai/

# On Pi
ssh pi@raspberrypi.local
cd tokkatot_ai/application
pip install -r requirements.txt
```

### Create systemd Service

```bash
sudo nano /etc/systemd/system/tokkatot-edge.service
```

Paste:
```ini
[Unit]
Description=Tokkatot Edge Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/tokkatot_ai/application
ExecStart=/usr/bin/python3 app.py --camera-id 0 --device cpu
Restart=on-failure
RestartSec=10
StandardOutput=journal

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tokkatot-edge
sudo systemctl start tokkatot-edge
```

Monitor:
```bash
sudo systemctl status tokkatot-edge
sudo journalctl -u tokkatot-edge -f
ps aux | grep python
top (watch CPU/memory)
```


---

## Cloud Integration

Edit `application/deployment/cloud_config.py`:

```python
CLOUD_CONFIG = {
    "ENABLED": True,
    "API_KEY": "your-api-key",
    "ENDPOINT": "https://cloud.tokkatot.ai/api/metrics",
    "DEVICE_ID": "pi-coop-1",
    "SEND_INTERVAL_MINUTES": 5,
}
```

Every 5 minutes, sends JSON with FPS, detections, disease rate, CPU, memory, uptime.

---

## Monitor App

```bash
# Check status
sudo systemctl status tokkatot-edge

# View logs
sudo journalctl -u tokkatot-edge -f

# Check running
sudo ps aux | grep python
top
```

---

## Troubleshoot

| Problem | Fix |
|---------|-----|
| App won't start | Check logs: `journalctl -u tokkatot-edge -n 50` |
| Camera not found | `ls /dev/video*` |
| Cloud not sending | Check API key, endpoint, network |
| High CPU | Reduce FPS or use GPU |
| Memory leak | Restart: `sudo systemctl restart tokkatot-edge` |

---

## Health Check Cron

Auto-restart if down:
```bash
crontab -e
*/30 * * * * systemctl is-active tokkatot-edge || systemctl start tokkatot-edge
```

---

## What Gets Sent to Cloud

Every 5 min:
```json
{
  "device_id": "pi-coop-1",
  "fps": 22.3,
  "detections": 457,
  "disease_rate": 5.04,
  "cpu": 65,
  "memory_mb": 512,
  "uptime_hours": 24.5
}
```

Tokkatot dashboard receives and displays real-time metrics.
