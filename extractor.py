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



app = FastAPI()


class Extractor:
  def __init__(self):
    self.content={}

  def instascrapper(self, url):
    # Initialize the ApifyClient with your API token
    client = ApifyClient("apify_api_WcarMXHXk3hdrFL08bv6DSKekaYyhb1HoSCr")

    # Prepare the Actor input
    run_input = {
        "addParentData": False,
        "directUrls": [
            url
        ],
        "enhanceUserSearchWithFacebookPage": False,
        "isUserReelFeedURL": False,
        "isUserTaggedFeedURL": False,
        "resultsLimit": 1,
        "resultsType": "details",
        "searchLimit": 1,
        "searchType": "hashtag"
    }
    # Run the Actor and wait for it to finish
    run = client.actor("shu8hvrXbJbY3Eb9W").call(run_input=run_input)
    insta={}
    images=[]
    vids=[]
    # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
      insta['caption']=item['caption']
      insta['hashtags']=item['hashtags']
      vids.append(item['videoUrl'])
      if(len(item['childPosts'])>0):
        for i in item['childPosts']:
          if(i['type']=="Image"):
            images.append(i['displayUrl'])
          else :
            vids.append(i['videoUrl'])

    insta["images"]=images
    insta["vids"]=vids

    self.content["insta"]=insta
    return

  def downloads(self):
    count=0
    imgs=[]
    vids=[]
    for item in self.content['insta']['images']:
      image_url = item
      response = requests.get(image_url)

      if response.status_code == 200:
          with open(f'image_{count}.jpg', "wb") as file:
              file.write(response.content)
          imgs.append(f'image_{count}.jpg')
          print("Image downloaded successfully!")
      else:
          print("Failed to download image.")

      count=count+1

    count=0

    for item in self.content['insta']['vids']:
        video_url = item
        response = requests.get(video_url, stream=True)

        if response.status_code == 200:
            with open(f'video_{count}.mp4', "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            vids.append(f'video_{count}.mp4')
            print("Video downloaded successfully!")
        else:
            print("Failed to download video.")

        count=count+1

    self.content["insta"]["imgpath"]=imgs
    self.content["insta"]["vidspath"]=vids

    return


  def s2t(self):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True
    )
    


    count=0
    audios=[]
    texts=[]
    for item in self.content["insta"]["vidspath"]:
          video = VideoFileClip(item)
          video.audio.write_audiofile(f"output_{count}.mp3")
          audios.append(f"output_{count}.mp3")
          count=count+1

    result = pipe(audios, batch_size=len(audios))
    for i in result:
      texts.append(i["text"])

    self.content['insta']["audiotext"]=texts
    return
  

@app.get("/extract")
def extract(url :  str = Query(..., title="Instagram Post URL")):
    ext=Extractor()
    ext.instascrapper(url)
    print("Your video has been extracted succesfully :)")
    ext.downloads()
    ext.s2t()
    return ext.content



# def fastapi_handler(req):
    
#     handler = Mangum(app)
#     return handler(req, None)

# firebase_function = functions.https.on_request(fastapi_handler)


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1" , port=8000, log_level="debug")
