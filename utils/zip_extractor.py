import zipfile
import os



def extract_zip(zip_path, extract_to):
    """
    Extracts the contents of a zip file to a specified directory.

    :param zip_path: Path to the zip file
    :param extract_to: Directory where the contents will be extracted
    """
    # Checking if the zip file exists
    if not os.path.exists(zip_path):
        print(f"Error: The file {zip_path} does not exist.")
        return

    # Checking if the destination directory exists, if not, create it
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    try:
        # Opening the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extracting all the contents
            zip_ref.extractall(extract_to)

            print(f"Contents extracted to: {extract_to}")
    except zipfile.BadZipFile:
        print(f"Error: The file {zip_path} is not a valid zip file.")



zip_file_path = "C:\\Users\\user\\Desktop\\action-segmentation-using-ms-tcn\\gtea_png.zip"
zip_file_path_1 = "C:\\Users\\user\\Desktop\\action-segmentation-using-ms-tcn\\xml_labels.zip"
output_directory = "C:\\Users\\user\\Desktop\\action-segmentation-using-ms-tcn\\data\\dataset"
output_directory_1 = "C:\\Users\\user\\Desktop\\action-segmentation-using-ms-tcn\\data\\xml_data"
extract_zip(zip_file_path, output_directory)
extract_zip(zip_file_path_1, output_directory_1)
