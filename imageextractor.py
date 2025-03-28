from typing import Union
from fastapi.responses import StreamingResponse
from fastapi import FastAPI
import uvicorn
from fastapi import FastAPI, Depends,Query
import os


from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests
# from google.colab.patches import cv2_imshow
import cv2
import os
from PIL import Image
import requests
from apify_client import ApifyClient
from moviepy import VideoFileClip
from PIL import Image
import requests
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from together import Together
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from transformers import AutoImageProcessor, ViTMAEModel



app = FastAPI()

class ProductImage:
  def __init__(self,prompts,frame_count=15):
    self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    self.frame_count=frame_count
    self.frames=[]
    self.prompts=prompts
    self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    self.model_imgs = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

  def frameextractor(self,vidurl,output_folder):
      # frame_skip = 10  # Extract every 10th frame
      frame_skip=self.frame_count
      frame_count = 0
      saved_count = 0
      count=0
      for item in vidurl:
            cap = cv2.VideoCapture(item)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    frame_filename = os.path.join(output_folder, f"frame_{count}_{saved_count:04d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    self.frames.append(frame_filename)
                    saved_count += 1

                frame_count += 1
            count += 1

  def prodimageextractor(self,input_folder,output_folder):
    # image=cv2.imread()
    count=0
    for vals in os.listdir(input_folder):
          # print(vals)
          image=cv2.imread(f"{input_folder}/{vals}")
          print(f"{input_folder}/{vals}")
          inputs = self.processor(text=self.prompts, images=[image] * len(self.prompts), padding="max_length", return_tensors="pt")
          # predict
          with torch.no_grad():
            outputs = self.model(**inputs)
          preds = outputs.logits.unsqueeze(1)

          for i in range(0,len(self.prompts)):
            logits_np = preds[i].squeeze().numpy()
            logits_min = logits_np.min()
            logits_max = logits_np.max()
            logits_norm = (((logits_np - logits_min) / (logits_max - logits_min) )*255).astype(np.uint8)

            # Generate coordinate grid
            x = np.linspace(-1, 1, logits_np.shape[1])  # X-coordinates
            y = np.linspace(-1, 1, logits_np.shape[0])  # Y-coordinates
            X, Y = np.meshgrid(x, y)
            contour = plt.contourf(X, Y, logits_norm, levels=20, cmap="viridis")
            plt.close()
            _, binary = cv2.threshold(logits_norm, contour.levels[-2], 250, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image=cv2.resize(image,(352,352))
            for contour in contours:

                    x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
                    if w>20 and h>20:
                        cropped_region = image[y:y+h, x:x+w]

                        # Save the cropped image (Optional)
                        cv2.imwrite(f"{output_folder}/cropped_image_{count}.jpg", cropped_region)

                    count=count+1
                # cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding box in RED

  def forward(self,inputs):
        # inputs.pop('image_name', None)

        outputs = self.model_imgs(**inputs)
        return torch.sum(outputs.last_hidden_state, dim=1)

  def clustering(self,output_folder,k=5):
    image=[]
    lists=[]
    for i in os.listdir(output_folder):
      lists.append(i)
      image.append(Image.open(f"{output_folder}/{i}"))

    # image1 = Image.open("/content/finalimage/cropped_image_15.jpg")


    inputs = self.image_processor(images=image, return_tensors="pt")
    # print(inputs)
    extracts = self.forward(inputs)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(extracts.detach().numpy())
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    dicts={}
    for i in range(0,k):
      dicts[i]=[]
    for i,j in enumerate(labels):
          dicts[j].append(lists[i])
    return dicts
  # print(i)

@app.post('/imgext')
def extractkeyword(keywords :  list = Query(..., title="Keywords"),vids:list =Query(...,title="Path of vids"),input : str=Query(...,title="Input dir"),output : str=Query(...,title="Output dir")):
    imgs=ProductImage(keywords)
    imgs.frameextractor(vids,output)
    imgs.prodimageextractor(output,input)
    dicts=imgs.clustering(input,6)
    return dicts




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1" , port=8002, log_level="debug")
