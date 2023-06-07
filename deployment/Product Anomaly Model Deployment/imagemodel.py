from pydantic import BaseModel
import base64
# 2. Class which describes Bank Notes measurements
class ImageModel(BaseModel):
    image: str