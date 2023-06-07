import pandas as pd
import requests
import os

# Path to the CSV file
csv_file_path = "data-collection/hardware_products.csv"

# Name of the column containing the image links
image_column = "poduct_image_url"

# Directory to save the downloaded images
save_directory = "data-collection/images"

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Load the CSV file using pandas
df = pd.read_csv(csv_file_path)

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    image_url = row[image_column]
    
    # Skip the row if the image URL is missing or invalid
    if pd.isna(image_url) or not isinstance(image_url, str):
        continue
    
    try:
        # Send a GET request to download the image
        response = requests.get(image_url)
        
        # Extract the filename from the image URL
        filename = os.path.basename(image_url)
        
        # Save the image to the specified directory
        save_path = os.path.join(save_directory, filename[2:-5])
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        print(f"Downloaded image: {filename}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from URL: {image_url}")
        print(f"Error message: {e}")