import requests
content = requests.get("https://raw.githubusercontent.com/Maverick341/cv/refs/heads/main/notebooks/Exp1.ipynb")

# Extract only the source field from each cell
notebook_data = content.json()
cells = notebook_data['cells']

# Print source content of each cell
for i, cell in enumerate(cells):
    print(f"Cell {i+1} ({cell['cell_type']}):")
    print("Source:")
    for line in cell['source']:
        print(line, end='')
    print("\n" + "="*50 + "\n")


# Alternative: Get all source content as a list
# all_sources = [cell['source'] for cell in cells]
# print("All source fields:")
# print(all_sources)