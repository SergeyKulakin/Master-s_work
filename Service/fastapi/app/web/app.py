from fastapi import FastAPI

from app.web.routes import router


app = FastAPI()
app.include_router(router)

