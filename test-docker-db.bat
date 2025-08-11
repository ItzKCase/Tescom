@echo off
echo 🐳 Testing Equipment Database in Docker Container
echo ================================================

echo 1. Building Docker image...
docker build -t tescom-agent-test .

if %ERRORLEVEL% neq 0 (
    echo ❌ Docker build failed
    exit /b 1
)

echo ✅ Docker build successful
echo.

echo 2. Testing database access in container...
docker run --rm -e OPENAI_API_KEY=test_key -e SERPER_API_KEY=test_key tescom-agent-test python test_docker_db.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ Database test PASSED - Ready for deployment!
) else (
    echo.
    echo ❌ Database test FAILED - Need to fix issues
    
    echo.
    echo 3. Running database initialization...
    docker run --rm -e OPENAI_API_KEY=test_key -e SERPER_API_KEY=test_key tescom-agent-test python docker-init-db.py
    
    if %ERRORLEVEL% equ 0 (
        echo.
        echo ✅ Database initialization successful!
        echo 🔄 Re-testing database access...
        
        docker run --rm -e OPENAI_API_KEY=test_key -e SERPER_API_KEY=test_key tescom-agent-test python test_docker_db.py
        
        if %ERRORLEVEL% equ 0 (
            echo.
            echo 🎉 All tests PASSED after initialization!
        ) else (
            echo.
            echo ❌ Database still not working after initialization
        )
    ) else (
        echo.
        echo ❌ Database initialization failed
    )
)

echo.
echo 4. Testing actual app startup...
start /b docker run --rm -e OPENAI_API_KEY=test_key -e SERPER_API_KEY=test_key -p 8080:7860 tescom-agent-test

timeout /t 10 /nobreak >nul

curl -f http://localhost:8080/ >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✅ App is responding to requests
) else (
    echo ⚠️  App may still be starting or not responding
)

echo.
echo 🏁 Docker testing complete!
echo To stop any running containers: docker stop $(docker ps -q --filter ancestor=tescom-agent-test)
