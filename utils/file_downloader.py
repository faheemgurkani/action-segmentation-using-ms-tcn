# # Redundant script
# import requests



# def download_file_from_google_drive(url, destination):
#     print(url)  # For, testing
    
#     # Sending the request
#     response = requests.get(url, stream=True)
    
#     if response.status_code == 200:
#         # Saving the file locally
#         with open(destination, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print(f"File downloaded successfully to {destination}")
#     else:
#         print("Failed to download the file.")



# # Replacing 'FILE_ID' with the actual file ID and specify the destination path
# url = "https://drive.google.com/drive/folders/1-MyUIACNlhFu0lppBO_PK9Uh7i77S9-7?usp=drive_link"
# destination = "C:\\Users\\user\\Desktop\\action-segmentation-using-ms-tcn\\data\\frames"
# download_file_from_google_drive(url, destination)
