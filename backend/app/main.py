from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult

from worker.tasks import recognize, add_face

app = FastAPI()


@app.post("/recognize/")
async def recognize_faces(files: list[UploadFile] = File(...)):
    """
    Submit multiple images to recognize faces.
    """
    tasks = []
    try:
        for file in files:
            try:
                contents = await file.read()
                task = recognize.delay(contents)
                tasks.append({
                    "filename": file.filename,
                    "task_id": task.id,
                    "status": "PROCESSING",
                    "url_result": f"/result/{task.id}",
                    "url_status": f"/status/{task.id}"
                })
            except Exception as e:
                tasks.append({
                    "filename": file.filename,
                    "task_id": None,
                    "status": "FAILED",
                    "error": str(e)
                })
        return JSONResponse(status_code=202, content=tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")


@app.post("/add_face/")
async def add_face_api(
    file: UploadFile = File(...),
    name: str = Form(...),
    user_id: str = Form(...)
):
    try:
        contents = await file.read()
        task = add_face.delay(contents, name, user_id)
        return {
            "task_id": task.id,
            "status": "PROCESSING",
            "url_result": f"/result/{task.id}",
            "url_status": f"/status/{task.id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit add_face task: {e}")


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    try:
        task = AsyncResult(task_id)
        if not task.ready():
            return {
                "task_id": task_id,
                "status": task.status,
                "result": None
            }

        result = task.result
        return {
            "task_id": task_id,
            "status": result.get("status"),
            "result": result.get("result")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching result: {e}")


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    try:
        task = AsyncResult(task_id)
        return {
            "task_id": task.id,
            "status": task.status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching task status: {e}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
