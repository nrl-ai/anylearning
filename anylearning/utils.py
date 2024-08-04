import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(base_path, contains=None):
    """List all images in a directory"""
    return list_files(base_path, valid_exts=image_types, contains=contains)


def list_files(basePath, valid_exts=None, contains=None):
    """List all files in a directory"""
    for root, _, filenames in os.walk(basePath):
        for filename in filenames:
            # If the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # Determine the file extension of the current file
            ext = filename[filename.rfind(".") :].lower()

            # Check to see if the file is an image and should be processed
            if valid_exts is None or ext.endswith(valid_exts):
                # Construct the path to the image and yield it
                image_path = os.path.join(root, filename)
                yield image_path
