import os
import uvicorn
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

if __name__ == "__main__":
    
    if os.getenv("MODE") != "dev":
        from backend.service.service import app
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        uvicorn.run("service.service:app", host="127.0.0.1", port=8000, reload=True)