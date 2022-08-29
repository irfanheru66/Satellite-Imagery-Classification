import cv2
import numpy as np
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import base64

class_names = ["Cloudy","Desert","Green area","Water"]
resultPath = 'static/result.jpg'
labels = []

def gridFind(img,gridSize,W):
    combination = [[i.min(), i.max()] for i in np.array_split(range(W),W//gridSize)]
    grid = []
    coor = []
    for j,jj in combination:
      for i,ii in combination:
        coor.append([[i,j],[ii,jj]])
        grid.append(img[j:jj,i:ii])
    return grid,coor

def plotDrawgrid(coors,drawedImg,grid,sz,model):
  drawedImg = cv2.cvtColor(drawedImg, cv2.COLOR_BGR2RGB)
  sz = sz // 4
  for i in range(len(coors)):
    start = coors[i][0]
    end = coors[i][1]
    label,conf = pred(grid[i],model)
    print(label)
    labels.append(label)
    cv2.rectangle(drawedImg,tuple(start),tuple(end),(0,0,0),5)
    if label == "cloudy":
      cv2.putText(drawedImg, 'Cloudy', (start[0]+sz,start[1]+sz), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
      cv2.putText(drawedImg, f'conf:{conf}', (start[0]+sz,start[1]+sz + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    elif label == "Desert":
      cv2.putText(drawedImg, 'Desert', (start[0]+sz,start[1]+sz), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
      cv2.putText(drawedImg, f'conf:{conf}', (start[0]+sz,start[1]+sz + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
    elif label == "Green Area":
      cv2.putText(drawedImg, 'Green_area', (start[0]+sz,start[1]+sz), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
      cv2.putText(drawedImg, f'conf:{conf}', (start[0]+sz,start[1]+sz + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
    elif label == "Water":
      cv2.putText(drawedImg, 'water', (start[0]+sz,start[1]+sz), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
      cv2.putText(drawedImg, f'conf:{conf}', (start[0]+sz,start[1]+sz + 50), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0,0,255), 2, cv2.LINE_AA)
  return drawedImg

def pred(img,model):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = Image.fromarray(img)

  tfms = transforms.Compose([transforms.Resize(64),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
  img = tfms(img).unsqueeze(0)
  model.eval()
  with torch.no_grad():
    output = model(img.to("cpu"))
    _, pred = torch.max(output, 1)

  return class_names[pred],str(round(output.tolist()[0][pred]))

def percent(count,len):
  return str((count/len)*100) + "%"

def main(src):

    img = cv2.imread(src)
    model = torch.load("mobilenetV3.pth",map_location=torch.device('cpu'))
    gridSize = 256

    drawedImg = img.copy()
    H, W = img.shape[:2]

    grid,coor = gridFind(img,gridSize,W)
    result = plotDrawgrid(coor,drawedImg,grid,gridSize,model)

    verdic_data ={}

    for name in class_names:
        verdic_data[name] = percent(labels.count(name),len(labels))
    #print(verdic_data)

    cv2.imwrite(resultPath,cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return verdic_data

def process64(src):
  npimg = np.frombuffer(src, dtype=np.uint8)
  img = cv2.imdecode(npimg,1)
  # cv2.imwrite("static/source.jpg",img)

  model = torch.load("mobilenetV3.pth",map_location=torch.device('cpu'))
  gridSize = 256

  drawedImg = img.copy()
  H, W = img.shape[:2]

  grid,coor = gridFind(img,gridSize,W)
  result = plotDrawgrid(coor,drawedImg,grid,gridSize,model)

  verdic_data ={}

  for name in class_names:
      verdic_data[name] = percent(labels.count(name),len(labels))
  # print(verdic_data)
  
  
  cv2.imwrite(resultPath,cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
  with open(resultPath, "rb") as image_file:
    result64 = base64.b64encode(image_file.read())
  verdic_data["image"] = str(result64)

  return verdic_data

