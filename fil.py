import pandas as pd
import torch.nn
import re
import requests
URL="your server url hosted"
HEADERS = {
    "xc-token": "your_api_token_here"
}
response = requests.get(URL, headers=HEADERS)
if response.status_code == 200:
    full_data = response.json()
    
    my_messages = full_data.get("list", [])
    
    print(f"Success! Found {len(my_messages)} messages.")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
def translate_to_numbers(text):
    text_lower = text.lower()
    
    f1 = 1 if re.search(r"https?|www|\.com|\.dz|link", text_lower) else 0
    
    scam_patterns = [
        r"fr[e3]{2}",
        r"n[1i]tro",
        r"fl[1i]xy",
        r"g[ir]atuit",
        r"cl[1i]ck",
    ]
    f2 = (
        1
        if any(re.search(pattern, text_lower) for pattern in scam_patterns)
        else 0
    )
    f3 = (
        sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
    )
    
    f4 = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    
    f5 = len(text) / 200  # Normalized
    
    return torch.tensor([[f1, f2, f3, f4, f5]], dtype=torch.float32)
all_tensors = []
all_labels = []

for row in my_messages:
    text = row.get("message", "")
    label = row.get("label")
    
    features = translate_to_numbers(text)
    
    all_tensors.append(features)
    all_labels.append(label)

print(f"Prepared {len(all_tensors)} items for the PyTorch model!")
X_train = torch.cat(all_tensors, dim=0)


y_train = torch.tensor([float(l) for l in all_labels]).view(-1, 1)

print("Final  ")
print(f"X_train Shape: {X_train.shape}")  
print(f"y_train Shape: {y_train.shape}")  
print("\nFirst row of data (Features):", X_train[0])
print("First label:", y_train[0])
