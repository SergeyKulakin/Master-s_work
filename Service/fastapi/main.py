import uvicorn

from app.web.app import app
from app.models.common import load_models


load_models()


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
