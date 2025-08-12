# 🏗️ Recommended Project Structure for Docker Deployment

## 📁 **Current vs. Recommended Layout**

### **❌ Don't Do This (Git-tracked database)**
```
Tescom/
├── equipment.db          ← ❌ Don't commit this to git
├── agent.py
├── app.py
├── docker-compose.yml
└── ...
```

### **✅ Do This Instead (Docker-managed database)**
```
Tescom/
├── app.py
├── agent.py
├── docker-compose.yml
├── Dockerfile
├── Tescom_new_list.xlsx      ← ✅ Source data (git-tracked)
├── Tescom_Functions.json     ← ✅ Configuration (git-tracked)
├── manufacturer_alias.csv     ← ✅ Configuration (git-tracked)
├── model_alias.csv           ← ✅ Configuration (git-tracked)
├── .gitignore               ← ✅ Excludes database files
└── ...
```

## 🔄 **How Data Flow Works**

### **1. Source Data (Git-tracked)**
- `Tescom_new_list.xlsx` - Your equipment list
- `Tescom_Functions.json` - Lab capabilities
- `manufacturer_alias.csv` - Name mappings
- `model_alias.csv` - Model mappings

### **2. Runtime Data (Docker-managed)**
- `equipment.db` - Generated database (in `/app/data/`)
- `logs/` - Application logs (in `/app/logs/`)
- `backups/` - Database backups (in `/app/backups/`)

## 🐳 **Docker Volume Strategy**

### **Persistent Volumes (Data that survives restarts)**
```yaml
volumes:
  - tescom-data:/app/data      # equipment.db
  - tescom-logs:/app/logs      # agent.log, capability_changes.log
  - tescom-backups:/app/backups # JSON backups
```

### **Read-only Mounts (Source files from git)**
```yaml
volumes:
  - ./Tescom_new_list.xlsx:/app/Tescom_new_list.xlsx:ro
  - ./Tescom_Functions.json:/app/Tescom_Functions.json:ro
  - ./manufacturer_alias.csv:/app/manufacturer_alias.csv:ro
  - ./model_alias.csv:/app/model_alias.csv:ro
```

## 🔧 **Database Initialization Process**

### **When Container Starts:**
1. **Check if** `/app/data/equipment.db` exists
2. **If missing**: Run `docker-init-db.py` to build from Excel
3. **If exists**: Use existing database

### **After Git Pull:**
1. **Stop container**: `docker-compose down`
2. **Rebuild**: `docker-compose build --no-cache`
3. **Start**: `docker-compose up -d`
4. **Database**: Automatically rebuilt from latest Excel data

## 📋 **Best Practices Checklist**

### **✅ Do These:**
- [ ] Keep source files in git (`Tescom_new_list.xlsx`, etc.)
- [ ] Use Docker volumes for runtime data
- [ ] Exclude database files from git (`.gitignore`)
- [ ] Mount source files as read-only in containers
- [ ] Use `docker-init-db.py` for database initialization

### **❌ Don't Do These:**
- [ ] Commit `equipment.db` to git
- [ ] Store database in project root
- [ ] Mount entire project directory into container
- [ ] Ignore volume persistence

## 🚀 **Workflow After Git Pull**

### **Option 1: Quick Update (Recommended)**
```bash
# After git pull
python update-docker-container.py
# Choose: 1 (Quick Update)
```

### **Option 2: Manual Update**
```bash
# After git pull
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### **What Happens:**
1. **Container stops** with old code
2. **New image builds** with updated source files
3. **Container starts** with new code
4. **Database rebuilds** from latest Excel data
5. **App runs** with latest changes

## 🔍 **Verification Commands**

### **Check Database Location:**
```bash
# Inside container
docker exec -it tescom-agent ls -la /app/data/
# Should show: equipment.db

# Check database records
docker exec -it tescom-agent python -c "
import sqlite3
conn = sqlite3.connect('/app/data/equipment.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM equipment;')
print(f'Records: {cursor.fetchone()[0]}')
conn.close()
"
```

### **Check Volume Mounts:**
```bash
docker inspect tescom-agent | grep -A 10 "Mounts"
```

## 💡 **Why This Approach is Superior**

### **1. Git Hygiene**
- Source data is version controlled
- Generated files stay out of git
- Clean commit history

### **2. Docker Best Practices**
- Follows 12-factor app principles
- Proper separation of concerns
- Production-ready deployment

### **3. Development Workflow**
- Easy to test changes
- Consistent environment
- No local database conflicts

### **4. Production Deployment**
- Same process works everywhere
- Data persistence across deployments
- Easy rollbacks

## 🎯 **Summary**

**Your current setup is PERFECT!** Moving `equipment.db` to `/app/data/` is exactly what you should do. This is the standard Docker pattern for:

- ✅ **Data persistence**
- ✅ **Git independence** 
- ✅ **Production readiness**
- ✅ **Easy updates**

**Keep doing exactly what you're doing!** 🎉
