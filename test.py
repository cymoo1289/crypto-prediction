
import pickle
from pathlib import Path

import streamlit_authenticator as stauth


names = ["moo", "cy"]
usernames = ["MOO", "CY"]

passwords = ["123", "abc"]

hashed_passwords = stauth.Hasher(passwords).generate()
print(hashed_passwords)
