# Satellite-Imagery-Classification

This program receive an Satellite image with 1024 x 1024 Resolution

<p float="left">
  <img src="readmeAssets/Satellite Photo 4 copy.jpg" width="300" />
  <img src="readmeAssets/result (10).jpg" width="300" /> 
</p>

# Run this app via Docker
## `irfanheru66/satellite-imagery-classification`

# To load the weight use Gdown to download the .pth
```
pip3 install gdown
gdown --fuzzy https://drive.google.com/file/d/1EBYQN9evN9lXOLaGqah29k7mOxlvcrrW/view?usp=sharing
```


# Docker Pull
```
docker pull irfanheru66/satellite-imagery-classification:1.0
```
# Docker Run
```
docker run -p 5000:5000 irfanheru66/satellite-imagery-classification:1.0
```
- after that, on your local browser clik localhost:5000

# You can use this app via HTML

<p float="left">
  <img src="readmeAssets/upload.png" width="400" />
  <img src="readmeAssets/result.png" width="400" /> 
</p>

# Or, you can use the api
```
localhost:5000/api
```
- Send image that are already being converted to Base64

```
{
    image:"Base64 Converted String"
}
```

- The return that you will get is

```
       {
        "Cloudy":,
        "Desert":,
        "Green area":,
        "Water":,
        "image":Base64Converted image


       }
```
