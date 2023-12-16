from typing import Optional

import cv2
import keras
import numpy as np
import uvicorn

from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware

from BackendModels.Cap import CapModel
from ScriptsMain.Firebase import Firebase
from ScriptsMain.DetectCaps import detect_caps
from ScriptsMain.Pinecone import PineconeContainer
from ScriptsMain.UtilsFun import img_to_numpy
from ScriptsMain.cnn import identify_cap, get_model, transform_imag_to_pinecone_format
from fastapi.responses import JSONResponse

app = FastAPI()
pinecone_container: PineconeContainer = PineconeContainer()
model: keras.Sequential = get_model()
firebase = Firebase()

origins = [
    "*",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_image(file_contents: bytes, user_id: str):
    image = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)
    image = img_to_numpy(image)
    cropped_images = detect_caps(image)
    caps_identified = []
    for cap in cropped_images:
        caps_identified.append(identify_cap(cap=np.array(cap[0]), model=model, pinecone_con=pinecone_container, user_id=user_id))
    positions = [tuple(int(v) for v in rct) for (img, rct) in cropped_images]

    result = {'positions': positions, 'caps_identified': caps_identified}
    print(caps_identified)
    for possible_values in caps_identified:
        for value in possible_values:
            value['image_url'] = firebase.get_image(f'users/{user_id}/bottle_caps/{value["metadata"]["name"]}.jpg')

    print(result)
    return result

def identify( user_id: str, file_path: str):
    with open(file_path, 'rb') as file:
        file_content = file.read()
    image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
    cap_identified = identify_cap(cap=np.array(image), model=model, pinecone_con=pinecone_container, user_id=user_id)
    #cap_identified = [cap.to_dict() for cap in cap_identified]
    return JSONResponse(cap_identified)

def detect(file_path: str):
    with open(file_path, 'rb') as file:
        file_content = file.read()
    image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
    image = img_to_numpy(image)
    cropped_images, marked_image = detect_caps(image)
    positions = [tuple(int(v) for v in rct) for (img, rct) in cropped_images]
    cv2.imshow("Detected Caps", marked_image)  # Display the marked image
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close all OpenCV windows
    return positions

def add_to_database(
        cap: CapModel,
        file_path: str):
    with open(file_path, 'rb') as file:
        file_content = file.read()
    image = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
    image = img_to_numpy(image)

    metadata = {
        'name': cap.name,
        'description': cap.description,
        'user_id': cap.user_id
    }

    cap_info = transform_imag_to_pinecone_format(model=model, img=image, metadata=metadata)
    pinecone_container.upsert_to_pinecone(cap_info=cap_info)
    return JSONResponse(cap_info)

# @app.post("/detect_and_identify")
# async def upload_file(user_id: str, file: UploadFile = File(...)):
#     result = process_image(await file.read(), user_id=user_id)
#     return JSONResponse(
#         content={"filename": file.filename,
#                  "positions": result['positions'],
#                  "caps": result['caps_identified']}
#     )


# @app.post("/detect")
# async def detect(file: UploadFile = File(...)):
#     image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
#     image = img_to_numpy(image)
#     cropped_images = detect_caps(image)
#     positions = [tuple(int(v) for v in rct) for (img, rct) in cropped_images]
#     return positions


# @app.post("/identify")
# async def identify( user_id: str, file: UploadFile = File(...)):
#     image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
#     cap_identified = identify_cap(cap=np.array(image), model=model, pinecone_con=pinecone_container, user_id=user_id)
#     cap_identified = [cap.to_dict() for cap in cap_identified]
#     return JSONResponse(cap_identified)


# @app.put("/add_to_database")
# async def add_to_database(
#         cap: CapModel = Depends(),
#         file: UploadFile = File(...),
# ):
#     image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
#     image = img_to_numpy(image)

#     metadata = {
#         'name': cap.name,
#         'description': cap.description,
#         'user_id': cap.user_id
#     }

#     cap_info = transform_imag_to_pinecone_format(model=model, img=image, metadata=metadata)
#     pinecone_container.upsert_to_pinecone(cap_info=cap_info)
#     return JSONResponse(cap_info)



if __name__ == '__main__':
    #uvicorn.run(app, host="localhost", port=8080)
    cap_tmp = CapModel(user_id='BetaTester', name='cap1', description='this is a test')
    # tmp = add_to_database(cap=cap_tmp, file_path="/Users/florianschiffel/Development/BottleCaps/database/caps-resized/9-maze_100.jpg")
    # tmp = detect("/Users/florianschiffel/Development/BottleCaps/database/test-images/test-i-have/white-background.jpg")
    tmp = identify("BetaTester", "/Users/florianschiffel/Development/BottleCaps/database/test-images/test-i-have/white-background.jpg")
    print(tmp)
