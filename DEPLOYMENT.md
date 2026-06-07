DEPLOYMENT_MD = """
# Deployment Guide

## Prerequisites

- Python 3.9+
- Docker and Docker Compose (for Milvus)
- 4GB+ RAM
- 10GB+ disk space

## Local Development

### 1. Clone and Setup

```bash
git clone <repository-url>
cd project_root
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 5. Start Milvus

```bash
docker-compose up -d
```

Verify Milvus is running:
```bash
docker-compose ps
```

### 6. Run Application

```bash
python main.py
```

API will be available at `http://localhost:8000`

## Production Deployment

### Option 1: Docker Deployment

#### Build Docker Image

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t data-analytics-api .
docker run -p 8000:8000 --env-file .env data-analytics-api
```

### Option 2: Cloud Deployment (AWS/GCP/Azure)

#### AWS EC2 Deployment

1. Launch EC2 instance (t3.medium or larger)
2. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip docker.io docker-compose
```

3. Clone repository and setup
4. Configure security groups (port 8000, 19530)
5. Run application with systemd service

Create `/etc/systemd/system/analytics-api.service`:
```ini
[Unit]
Description=Data Analytics API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/project_root
Environment="PATH=/home/ubuntu/project_root/venv/bin"
ExecStart=/home/ubuntu/project_root/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable analytics-api
sudo systemctl start analytics-api
```

### Option 3: Kubernetes Deployment

Create `k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analytics-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: analytics-api
  template:
    metadata:
      labels:
        app: analytics-api
    spec:
      containers:
      - name: api
        image: your-registry/analytics-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: gemini-key
---
apiVersion: v1
kind: Service
metadata:
  name: analytics-api-service
spec:
  selector:
    app: analytics-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

## Configuration for Production

### 1. Update app_config.yaml

```yaml
app:
  debug: false
  host: "0.0.0.0"
  port: 8000
```

### 2. Setup HTTPS

Use nginx as reverse proxy:
```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Setup Monitoring

Install Prometheus and Grafana for monitoring.

### 4. Setup Logging

Configure centralized logging (ELK stack, CloudWatch, etc.)

## Scaling Considerations

### Horizontal Scaling

1. Use external Milvus cluster
2. Implement Redis for session storage
3. Use load balancer (nginx, AWS ALB)
4. Stateless application design

### Vertical Scaling

- Increase RAM for larger datasets
- Use GPU for faster embedding generation
- SSD storage for Milvus

## Backup and Recovery

### Backup Milvus Data

```bash
docker-compose stop milvus-standalone
tar -czf milvus-backup-$(date +%Y%m%d).tar.gz volumes/milvus
docker-compose start milvus-standalone
```

### Restore from Backup

```bash
docker-compose stop milvus-standalone
tar -xzf milvus-backup-YYYYMMDD.tar.gz
docker-compose start milvus-standalone
```

## Troubleshooting

### Milvus Connection Issues

```bash
# Check Milvus logs
docker-compose logs milvus-standalone

# Restart Milvus
docker-compose restart milvus-standalone
```

### API Not Responding

```bash
# Check application logs
tail -f logs/app.log

# Verify port availability
netstat -tulpn | grep 8000
```

### Memory Issues

```bash
# Monitor memory usage
docker stats

# Increase Docker memory limit
# Edit docker-compose.yml and add:
    mem_limit: 4g
```

## Security Checklist

- [ ] API keys stored securely (not in code)
- [ ] HTTPS enabled
- [ ] Rate limiting implemented
- [ ] Input validation on all endpoints
- [ ] CORS configured properly
- [ ] Firewall rules configured
- [ ] Regular security updates
- [ ] Monitoring and alerting setup
"""

print("=" * 80)
print("ALL FILES CREATED SUCCESSFULLY!")
print("=" * 80)
print("\nTo use this refactored code:")
print("\n1. Create the directory structure as shown above")
print("2. Copy each code section to its respective file")
print("3. Install dependencies: pip install -r requirements.txt")
print("4. Setup .env file with your API keys")
print("5. Start Milvus: docker-compose up -d")
print("6. Run the application: python main.py")
print("\nSee README.md, API_DOCUMENTATION.md, and DEPLOYMENT.md for details")
"""