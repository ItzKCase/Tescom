# 🔧 Docker Troubleshooting Guide

## Equipment Database Not Working in Docker

### Quick Diagnosis Commands

```bash
# 1. Check if database exists in container
docker exec tescom-agent ls -la /app/data/equipment.db
docker exec tescom-agent ls -la /app/equipment.db

# 2. Test database connectivity
docker exec tescom-agent python test_docker_db.py

# 3. Check environment variables
docker exec tescom-agent env | grep -E "(DB_PATH|LOG_DIR)"

# 4. Check container logs for database errors
docker-compose logs tescom-agent | grep -i "database\|error\|sqlite"
```

### Common Issues & Solutions

#### Issue 1: Database File Not Found

**Symptoms:**
- App starts but shows "equipment not found" for all queries
- Error messages about missing database file

**Diagnosis:**
```bash
docker exec tescom-agent ls -la /app/data/
docker exec tescom-agent ls -la /app/equipment.db
```

**Solutions:**

**Option A: Build database in container**
```bash
# Build database from Excel file
docker exec tescom-agent python docker-init-db.py
```

**Option B: Copy database from host**
```bash
# Copy database to running container
docker cp equipment.db tescom-agent:/app/data/equipment.db

# Restart container to pick up changes
docker-compose restart tescom-agent
```

**Option C: Mount database volume**
```yaml
# In docker-compose.yml, add host mount
volumes:
  - ./equipment.db:/app/data/equipment.db:ro  # Read-only mount
  - tescom-logs:/app/logs
  - tescom-backups:/app/backups
```

#### Issue 2: Database Permissions

**Symptoms:**
- Database file exists but cannot be opened
- SQLite permission errors

**Diagnosis:**
```bash
docker exec tescom-agent ls -la /app/data/equipment.db
docker exec tescom-agent whoami
```

**Solution:**
```bash
# Fix permissions
docker exec tescom-agent chown appuser:appuser /app/data/equipment.db
docker exec tescom-agent chmod 644 /app/data/equipment.db
```

#### Issue 3: Environment Variables Not Set

**Symptoms:**
- App looks for database in wrong location
- Using default paths instead of Docker paths

**Diagnosis:**
```bash
docker exec tescom-agent env | grep DB_PATH
```

**Solution:**
```bash
# Check docker-compose.yml has correct environment variables
environment:
  - DB_PATH=/app/data/equipment.db
  - LOG_DIR=/app/logs
```

#### Issue 4: Volume Mount Issues

**Symptoms:**
- Database changes don't persist
- Fresh database on each restart

**Diagnosis:**
```bash
docker volume ls | grep tescom
docker volume inspect tescom_tescom-data
```

**Solution:**
```bash
# Recreate volumes
docker-compose down -v
docker-compose up -d
```

### Complete Rebuild Process

If all else fails, here's a complete rebuild:

```bash
# 1. Stop and remove everything
docker-compose down -v
docker rmi tescom-agent

# 2. Clean up volumes
docker volume prune -f

# 3. Rebuild from scratch
docker-compose build --no-cache

# 4. Initialize database
docker-compose up -d
docker-compose exec tescom-agent python docker-init-db.py

# 5. Test functionality
docker-compose exec tescom-agent python test_docker_db.py
```

### Testing Database Functionality

**1. Quick Test:**
```bash
# Test if app recognizes known equipment
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "keysight 34401a"}'
```

**2. Comprehensive Test:**
```bash
# Run full diagnostic
docker exec tescom-agent python test_docker_db.py
```

**3. Database Content Check:**
```bash
# Check database contents
docker exec tescom-agent sqlite3 /app/data/equipment.db "SELECT COUNT(*) FROM equipment;"
docker exec tescom-agent sqlite3 /app/data/equipment.db "SELECT manufacturer, model FROM equipment LIMIT 5;"
```

### Manual Database Setup

If automatic database setup fails:

```bash
# 1. Access container shell
docker exec -it tescom-agent bash

# 2. Check Excel file exists
ls -la Tescom_new_list.xlsx

# 3. Build database manually
python build_equipment_db.py --excel ./Tescom_new_list.xlsx --db /app/data/equipment.db --rebuild --report

# 4. Test the built database
python test_docker_db.py

# 5. Exit container
exit
```

### Production Deployment Checklist

- [ ] Equipment database exists and has data (>10,000 records)
- [ ] Environment variables properly set
- [ ] Volumes configured for data persistence
- [ ] API keys configured
- [ ] Health checks passing
- [ ] App responds to test queries
- [ ] Logs show no database errors

### Monitoring Commands

```bash
# Watch logs in real-time
docker-compose logs -f tescom-agent

# Check container health
docker ps | grep tescom-agent

# Monitor resource usage
docker stats tescom-agent

# Check disk usage
docker exec tescom-agent df -h
docker exec tescom-agent du -sh /app/data/
```

### Emergency Recovery

If the app is completely broken:

```bash
# 1. Stop everything
docker-compose down

# 2. Backup any important data
docker run --rm -v tescom_tescom-data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz -C /data .

# 3. Fresh start with working local database
cp equipment.db data-backup/
docker-compose up -d
docker cp data-backup/equipment.db tescom-agent:/app/data/

# 4. Restart to pick up database
docker-compose restart tescom-agent
```

### Getting Help

If issues persist, collect this information:

```bash
# System info
docker version
docker-compose version

# Container status
docker ps -a | grep tescom

# Logs
docker-compose logs tescom-agent > troubleshooting-logs.txt

# Database diagnostic
docker exec tescom-agent python test_docker_db.py > diagnostic-output.txt

# Environment
docker exec tescom-agent env > container-env.txt
```
