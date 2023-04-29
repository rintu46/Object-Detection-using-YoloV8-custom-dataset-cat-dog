from fastapi import FastAPI, File, UploadFile
import uvicorn
import fastapi
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
import io
import cvlib as cv
from cvlib.object_detection import draw_bbox
from fastapi.responses import StreamingResponse




from ultralytics import YOLO

print(fastapi.__version__)

IMAGEDIR = "Images/"

app  = FastAPI()

# load our trained model
Model = YOLO('best.pt')



@app.post("/predict")
async def predict(
    file: UploadFile = File()
):
    # Validate Input file
    filename = file.filename
    
    # Read image as a bytes
    image_stream = io.BytesIO(file.file.read())

    # Write the bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)

    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

     
    
    # Run the loaded model
    bbox, label, conf = cv.detect_common_objects(image, model=Model)
    
    # Create image that includes bounding boxes and labels
    output_image = draw_bbox(image, bbox, label, conf)
    
    # Save it in a folder within the server
    cv2.imwrite(f'Images/{filename}', output_image)
    
    
 
    
    # Open the saved image for reading in binary mode
    file_image = open(f'Images/{filename}', mode="rb")
    
    # Return the image as a stream specifying media type
    return StreamingResponse(file_image, media_type="image/jpeg")




if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)