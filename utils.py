import os
import errno


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
