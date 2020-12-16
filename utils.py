import os
import errno
import requests

def ensure_dir(filename):
  if not os.path.exists(os.path.dirname(filename)):
    try:
      os.makedirs(os.path.dirname(filename))
      return True

    except OSError as e:
      if e.errno != errno.EEXIST:
        raise

  else:
    return False

def download(item):
  url = item[0]
  fn = item[1]

  if not os.path.exists(fn):
    ensure_dir(fn)
    try:
      urllib.request.urlretrieve(url, fn)
    except:
      print('error while retrieving {:}'.format(url))
  else:
    pass
    #print('Skipped ', fn.split('/')[-1])
