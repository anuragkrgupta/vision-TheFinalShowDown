import glob

def check_classes():
    classes = set()
    txt_files = glob.glob('dataset/Door, Windows and Stairs Dataset/images/*.txt')
    print(f"Found {len(txt_files)} text files.")
    
    for f in txt_files:
        with open(f, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if parts:
                    classes.add(parts[0])
                    
    print(f"Unique class IDs found: {classes}")

if __name__ == '__main__':
    check_classes()
