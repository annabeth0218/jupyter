import os
import re
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup

# setup: session, id, dir ––––––––––––––––––––––––––––––––––
session = requests.Session()
login_url = "http://vgh.pathology.tw/ndp/serve/login" 
payload = {
    'username': 'lcw',
    'password': '5406c'
}
'''
STAGE_DIR = "hamamatsu"
slide_id = "S114-59685"  
save_path = f"{STAGE_DIR}/images/{slide_id}.jpg"

# login
response = session.post(login_url, data=payload)
if response.status_code == 200:
    print("Login successful!")
else:
    print("Login failed.")

# search
search_url = f"http://vgh.pathology.tw/ndp/serve/search"
params = {'q': patient_id}
response = session.get(search_url, params=params)

soup = BeautifulSoup(response.text, 'html.parser')
objectid = None
for link in soup.find_all('a', href=True):
    if "objectid=" in link['href']:
        # Extract the GUID from the URL
        parts = link['href'].split('objectid=')
        if len(parts) > 1:
            objectid = parts[1].split('&')[0]
            break

guid_pattern = r'[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}'
matches = re.findall(guid_pattern, response.text)
print(matches)


print(f"Target ObjectID found: {objectid}")
'''
image_url = f"http://vgh.pathology.tw/ndp/serve/view?contextid=&objectid=7C7F3CE9-DE81-465F-84A9-5762AABA40B7"

try:
    response = session.get(image_url, timeout=30)
    response.raise_for_status() 
    image = Image.open(BytesIO(response.content))
    image.save(save_path)
    print(f"Image Size: {image.size}")
# -
except Exception as e:
    print(f"Failed to download image: {e}")


import requests
from PIL import Image
from io import BytesIO

session = requests.Session()
login_url = "http://vgh.pathology.tw/ndp/serve/login"
payload = {'username': 'lcw', 'password': '5406c'}

# 1. Actually log in
login_response = session.post(login_url, data=payload)

# 2. Define your save path
save_path = "slide_screenshot.png"

# 3. Note: This URL must point to a RENDERED image, not a viewer page
image_url = "http://vgh.pathology.tw/ndp/serve/view?contextid=&objectid=7C7F3CE9-DE81-465F-84A9-5762AABA40B7"

try:
    response = session.get(image_url, timeout=30)
    response.raise_for_status() 
    
    # Check if the response is actually an image
    if "text/html" in response.headers.get("Content-Type", ""):
        print("Error: The URL returned a webpage, not an image file.")
    else:
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        print(f"Image saved! Size: {image.size}")

except Exception as e:
    print(f"Failed: {e}")

    