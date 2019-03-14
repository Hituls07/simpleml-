import os
from six.moves import urllib
import tarfile


def fetch_data(url, filename, download_location = os.getcwd()):
    """
    :param url: URL where file is lying
    :param filename: file name with extension
    :param download_location: Location to download
    :return:
    """
    # Checks whether download location exist or not
    if not os.path.isdir(download_location):
        os.makedirs(download_location)  # else make it
    # Joining url and file name
    file_path = os.path.join(download_location, filename)
    url_path = url + '/' + filename
    # Retrieving data
    urllib.request.urlretrieve(url_path, file_path)
    # Extract data if file is in tar format
    if filename.split('.')[1] == 'tgz':
        tgz = tarfile.open(file_path)
        tgz.extractall(download_location)
        tgz.close
    print('File has been uploaded')


