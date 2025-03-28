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


class Product:
  def __init__(self,keyinput):
    self.keyinput=keyinput
    self.max_tokens=500
    self.temperature=0.7
    self.top_p=0.7
    self.top_k=50
    self.repetition_penalty=1
    self.stop=["<|eot_id|>","<|eom_id|>"]
    self.stream=True
    self.model="meta-llama/Llama-3.3-70B-Instruct-Turbo"
    self.keylist=""
    self.head=""
    self.description=""
    self.keyword=[{
                "role": "system",
                "content": f"""Extract important keywords from the paragraph below that describe the product being mentioned. Each keyword should be separated by a comma (,). Do not include descriptive adjectives or repetitive words with similar meanings. The keywords should focus on the product's characteristics, features, or unique aspects. Only generate the keywords as output, without any additional text.
                Paragraph : {self.keyinput}
                Output : """
        }]
    self.heading=[{
                "role": "system",
                "content": f"""Your task is to generate heading for the product described in the paragraph mentioned below.The product heading must be optimized for Amazon's SEO. Never use any irrelevant information apart from mentioned in the paragraph. Some keywords will be given for example you can use them or any similar keywords. Follow the rules below to create a good heading.
                The output should only be the the product heading.
                Rules :
                Amazon allows sellers 250 characters or your product title.

                They also recommend you use a formula when creating your title.

                Headline Formula

                Product Brand/Description + Line/Collection + Material/Ingredient + Color/Size + Quantity

                Product Title example
                Your particular product may not need to include everything in the formula above.

                A good rule of thumb is without stuffing your product title, include all the information you think your customers need to make a buying decision and try to include keywords based on the paragraph.
                Paragraph : {self.keyinput}
                Output : """
        }]
    self.desc=[{
        "role": "system",
        "content":
        f"""
        Your task is to generate description for the product mentioned in the paragraph given below. Also there is a list of rules that must be followed while creating the description.Below is the set of rules which you must adhere to.Give only the product description as output.
        Rules:
        Include only product-related information, and don’t add misleading facts. Be transparent and honest about the product's features, benefits, and limitations. This will build trust with customers, as well as reduce the likelihood of returns or negative reviews.
Consider the character limit. On Amazon, you can use up to 2,000 characters to tell customers about your product and its features. As long as it fits within these parameters, take advantage of this text length, focusing on explaining the product’s characteristics and benefits.
Write clearly and concisely. Avoid using jargon or technical terms, unless they are essential for understanding the product. Organize information logically, using headings or paragraphs to break up content and improve readability. Additionally, don’t overdo it with your formatting, keeping in mind Amazon doesn’t favor HTML code in descriptions.
Testimonials or quotes of any kind are not allowed. Focus on objective and factual information, as Amazon's guidelines prohibit the use of any testimonials in product descriptions.
Here are some additional recommendations to accompany Amazon's rules.
Don’t include pricing information, as it will quickly become outdated. Instead, reserve that information for the pricing section or special offers section of your listing.
Include relevant keywords naturally in your description. Keywords should flow naturally, making the text informative and helping to increase your item’s search ranking. You also need to avoid keyword stuffing, especially in copywriting, as this can lead to penalties from Amazon.
       Paragraph: {self.keyinput}
       Output:
        """
    }]


  def keywordextractor(self,query,max_tokens=100):
     client = Together(api_key="63b1d2173ee710215020e9fe062a851308f26eb57ee32e62b4098cddbfe2112b")
     self.keyword.append({"role":"user", "content":query})
     response = client.chat.completions.create(
          model=self.model,
          messages=self.keyword,
          max_tokens=self.max_tokens,
          temperature=self.temperature,
          top_p=self.top_p,
          top_k=self.top_k,
          repetition_penalty=self.repetition_penalty,
          stop=self.stop,
          stream=self.stream
     )
     assist=""
     for token in response:
          if hasattr(token, 'choices'):
              assist+=token.choices[0].delta.content
              print(token.choices[0].delta.content, end='', flush=True)

     self.keylist=assist
     self.keyword.append({"role":"assistant","content":assist})

  def headingextractor(self,query,max_tokens=200):
     client = Together(api_key="63b1d2173ee710215020e9fe062a851308f26eb57ee32e62b4098cddbfe2112b")
     self.heading.append({"role":"user", "content":query})
     response = client.chat.completions.create(
          model=self.model,
          messages=self.heading,
          max_tokens=self.max_tokens,
          temperature=self.temperature,
          top_p=self.top_p,
          top_k=self.top_k,
          repetition_penalty=self.repetition_penalty,
          stop=self.stop,
          stream=self.stream
     )
     assist=""
     for token in response:
          if hasattr(token, 'choices'):
              assist+=token.choices[0].delta.content
              print(token.choices[0].delta.content, end='', flush=True)


     self.head=assist
     self.heading.append({"role":"assistant","content":assist})

  def descriptionextractor(self,query,max_tokens=10000):
     client = Together(api_key="63b1d2173ee710215020e9fe062a851308f26eb57ee32e62b4098cddbfe2112b")
     self.desc.append({"role":"user", "content":query})
     response = client.chat.completions.create(
          model=self.model,
          messages=self.desc,
          max_tokens=self.max_tokens,
          temperature=self.temperature,
          top_p=self.top_p,
          top_k=self.top_k,
          repetition_penalty=self.repetition_penalty,
          stop=self.stop,
          stream=self.stream
     )
     assist=""
     for token in response:
          if hasattr(token, 'choices'):
              assist+=token.choices[0].delta.content
              print(token.choices[0].delta.content, end='', flush=True)

     self.description=assist
     self.desc.append({"role":"assistant","content":assist})

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

# def initilization():
      
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.get("/extract")
def extract(url :  str = Query(..., title="Instagram Post URL")):
    ext=Extractor()
    ext.instascrapper(url)
    print("Your video has been extracted succesfully :)")
    ext.downloads()
    ext.s2t()
    return ext.content

@app.get("/keyword")
def extractkeyword(userquestion :  str = Query(..., title="Caption + Audio Text")):
    pro=Product(userquestion)
    pro.keywordextractor("Follow all the instruction properly.")
    print("\nKeywords have been generated")
    pro.headingextractor("Follow all the instruction properly.")
    print("\nHeading has been generated")
    pro.descriptionextractor("Follow all the instruction properly.")
    return {"keyword":pro.keylist,"heading":pro.head,"desc":pro.description}

@app.get('/desc')
def extractkeyword(userquestion :  str = Query(..., title="Caption + Audio Text"),descques:str =Query(...,title="Description related question")):
    pro=Product(userquestion)
    pro.descriptionextractor(descques)
    return {"desc":pro.description}

@app.get('/head')
def extractkeyword(userquestion :  str = Query(..., title="Caption + Audio Text"),descques:str =Query(...,title="Heading related question")):
    pro=Product(userquestion)
    pro.headingextractor(descques)
    return {"heading":pro.head}


@app.post('/imgext')
def extractkeyword(keywords :  list = Query(..., title="Keywords"),vids:list =Query(...,title="Path of vids"),input : str=Query(...,title="Input dir"),output : str=Query(...,title="Output dir")):
    imgs=ProductImage(keywords)
    imgs.frameextractor(vids,output)
    imgs.prodimageextractor(output,input)
    dicts=imgs.clustering(input,6)
    return dicts

# def fastapi_handler(req):
    
#     handler = Mangum(app)
#     return handler(req, None)

# firebase_function = functions.https.on_request(fastapi_handler)


# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1" , port=8000, log_level="debug")
