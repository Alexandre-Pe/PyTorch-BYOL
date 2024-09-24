import os

def preprocess_mpiifacegaze(path):
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory.")
    if os.path.exists(os.path.join(path, "images")):
        raise ValueError(f"{path} has already been preprocessed.")
    os.makedirs(os.path.join(path, "images"))
    os.makedirs(os.path.join(path, "labels"))
    for i in range(15):
        # Update image paths in labels
        with open(os.path.join(path, f"p{i:02d}", f"p{i:02d}.txt"), "r") as f:
            lines = f.readlines()
        with open(os.path.join(path, f"p{i:02d}", f"p{i:02d}.txt"), "w") as f:
            for line in lines:
                line = line.strip().split(" ")
                line[0] = os.path.join(f"p{i:02d}", line[0])
                f.write(" ".join(line) + "\n")
        # Move labels "p00.txt", "p01.txt", ..., "p14.txt" to "labels"
        os.rename(os.path.join(path, f"p{i:02d}", f"p{i:02d}.txt"), os.path.join(path, "labels", f"p{i:02d}.txt"))
        # Move directory "p00", "p01", ..., "p14" to "images"
        os.rename(os.path.join(path, f"p{i:02d}"), os.path.join(path, "images", f"p{i:02d}"))