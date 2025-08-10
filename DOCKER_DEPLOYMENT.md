# 🐳 Docker Deployment Guide

This guide explains how to deploy the Tescom Agent web application using Docker.

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file in your project root:

```bash
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here

# Optional: External logo URL
TESCOM_LOGO_URL=https://example.com/logo.png
```

### 2. Deploy with Docker Compose (Recommended)

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

The application will be available at `http://localhost:8080`

### 3. Deploy with Docker CLI

```bash
# Build the image
docker build -t tescom-agent .

# Run the container
docker run -d \
  --name tescom-agent \
  -p 8080:7860 \
  -v tescom-data:/app/data \
  -v tescom-logs:/app/logs \
  -v tescom-backups:/app/backups \
  -e OPENAI_API_KEY=your_key_here \
  -e SERPER_API_KEY=your_serper_key \
  tescom-agent
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *Required* | OpenAI API key for GPT-4o mini |
| `SERPER_API_KEY` | *Required* | Serper API key for web search |
| `GRADIO_HOST` | `0.0.0.0` | Host to bind Gradio server |
| `GRADIO_PORT` | `7860` | Port for Gradio server |
| `LOG_DIR` | `/app/logs` | Directory for log files |
| `DB_PATH` | `/app/data/equipment.db` | SQLite database path |
| `BACKUPS_DIR` | `/app/backups` | Directory for backup files |
| `TESCOM_LOGO_URL` | *None* | External URL for logo |
| `TESCOM_LOGO_PATH` | `/app/assets/tescom-logo.png` | Local logo file path |

### Data Persistence

The following directories are persisted using Docker volumes:
- `/app/data` - SQLite database and equipment data
- `/app/logs` - Application and capability change logs  
- `/app/backups` - Equipment list backups

## 📊 Monitoring

### Health Checks

The container includes built-in health checks:
```bash
# Check container health
docker ps
# Look for "healthy" status

# Manual health check
curl http://localhost:8080/
```

### Logs

```bash
# View application logs
docker-compose logs -f tescom-agent

# View specific log files
docker exec tescom-agent tail -f /app/logs/agent.log
docker exec tescom-agent tail -f /app/logs/capability_changes.log
```

## 🔄 Updates and Maintenance

### Updating the Application

```bash
# Pull latest code changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Database Management

```bash
# Access the database
docker exec -it tescom-agent sqlite3 /app/data/equipment.db

# Backup database
docker exec tescom-agent sqlite3 /app/data/equipment.db .dump > backup.sql

# View equipment data
docker exec tescom-agent ls -la /app/data/
```

### Volume Management

```bash
# List volumes
docker volume ls

# Backup volume data
docker run --rm -v tescom-data:/data -v $(pwd):/backup alpine tar czf /backup/tescom-data-backup.tar.gz -C /data .

# Restore volume data
docker run --rm -v tescom-data:/data -v $(pwd):/backup alpine tar xzf /backup/tescom-data-backup.tar.gz -C /data
```

## 🛠️ Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Change the host port in docker-compose.yml
   ports:
     - "8081:7860"  # Use port 8081 instead
   ```

2. **Permission issues**
   ```bash
   # Check container logs
   docker-compose logs tescom-agent
   
   # Fix volume permissions
   docker exec -it tescom-agent chown -R appuser:appuser /app/data /app/logs /app/backups
   ```

3. **API key issues**
   ```bash
   # Verify environment variables
   docker exec tescom-agent env | grep API_KEY
   ```

4. **Database connection issues**
   ```bash
   # Check database file
   docker exec tescom-agent ls -la /app/data/equipment.db
   
   # Test database connectivity
   docker exec tescom-agent python -c "import sqlite3; sqlite3.connect('/app/data/equipment.db').execute('SELECT 1')"
   ```

### Development Mode

For development with live code updates:

```bash
# Mount source code directory
docker run -d \
  --name tescom-agent-dev \
  -p 8080:7860 \
  -v $(pwd):/app \
  -v tescom-data:/app/data \
  -e OPENAI_API_KEY=your_key \
  tescom-agent
```

## 🔐 Security Notes

- The container runs as non-root user (`appuser`) for security
- Sensitive data (API keys) should be passed via environment variables
- Use Docker secrets in production for sensitive information
- Regular security updates: `docker-compose pull && docker-compose up -d`

## 📈 Production Deployment

For production deployment, consider:

1. **Reverse Proxy**: Use nginx or Traefik for SSL termination
2. **Resource Limits**: Set CPU/memory limits in docker-compose.yml
3. **Monitoring**: Add Prometheus metrics and health check endpoints
4. **Backup Strategy**: Automate database and volume backups
5. **Log Management**: Forward logs to centralized logging system
