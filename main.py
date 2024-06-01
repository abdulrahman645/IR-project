from fastapi import FastAPI
from text_processing.text_processing import router as text_processing_router
from indexing.indexing import router as indexing_router
from search.search import  router as search_router


app = FastAPI()

app.include_router(text_processing_router)
app.include_router(indexing_router)
app.include_router(search_router)