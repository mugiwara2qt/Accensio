import uvicorn
import os

if __name__ == "__main__":
    print("==================================================")
    print("  LAUNCHING ACCENSIO SPACE FIRE SUPER INTELLIGENCE  ")
    print("==================================================")
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=False)