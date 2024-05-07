'''Data preprocessing script to download the SEC 10-K filings for Apple and Pepsico 
    and convert to usable format'''

import os
import glob
import shutil
from sec_edgar_downloader import Downloader

# Download the last 29 SEC 10-K filings for Apple, Pepsico, Goldman Sachs
dl = Downloader("User-Agent", "abbinavsankar2003@gmail.com")
equity_ids = ["AAPL", "PEP"]
for equity_id in equity_ids:
    dl.get("10-K", equity_id, after = "1995-1-1", before = "2024-3-1")

# Function to copy only the sec filings to the data folder
def copy_files(path: str):
    idx = 0
    os.mkdir("./data")
    TextFiles = glob.glob(path)
    for file in TextFiles:
        idx += 1
        shutil.copy(file, "./data/" + f"File_{idx}" + ".txt")
        
    return str("Successfully Copied " + str(idx) + " files !!")

print(copy_files("sec-edgar-filings/*/10-K/*/*.txt"))
