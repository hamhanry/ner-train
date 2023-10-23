import re
from collections import Counter

def basic_preprocessing(text):
  text=text.lower()
  text=re.sub(r'[^\w\s]','',text)
  text = re.sub(r'@\w+', '', text)
  return text

def remove_duplicates(input):

    # split input string separated by space
    input = input.split(" ")

    # now create dictionary using counter method
    # which will have strings as key and their
    # frequencies as value
    UniqW = Counter(input)

    # joins two adjacent elements in iterable way
    s = " ".join(UniqW.keys())
    return s