import argparse
import re
from pathlib import Path

classnames_txt = "./datasets/ImageNet/classnames.txt"
ouput_dir = "./datasets/ImageNet/imagenet_"

def glob_re(pattern, strings):
    return filter(lambda s: re.compile(pattern).match(str(s)), strings)

def read_classnames():
    classnames = []
    with open(classnames_txt, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(" ")
            folder = line[0]
            classnames.append(folder)
    return classnames

def main(args):
    classnames = read_classnames()
    uqclassnames = set(classnames)
    cls2idx = {cls: idx for idx, cls in enumerate(classnames)}
    uqclassidx = {v:k for k, v in cls2idx.items()}

    data_dir = Path(args.data_dir)
    jpgimages = glob_re(r'.*\.(?:JPEG|jpeg|jpg)$', (data_dir/args.dataset).glob("**/*"))
    out = str(Path(ouput_dir)) + args.split

    with open(f"{out}.txt", "w") as f:
        for img in jpgimages:
            if img.is_file():
                cls = img.parent.name
                if args.split == "v2":
                    if int(cls) in uqclassidx:
                        label = int(cls)
                        f.write(f"{str(img.relative_to(data_dir))}{args.delimiter}{label}\n")
                else:
                    if cls in uqclassnames:
                        label = cls2idx[cls]
                        f.write(f"{str(img.relative_to(data_dir))}{args.delimiter}{label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet split")
    parser.add_argument("--dataset", type=str, default="imagenet",
                        help="Dataset name")
    parser.add_argument("--data_dir", type=str, default="/projets/masih/dataset/imagenet")
    parser.add_argument("--delimiter", type=str, default=" ")
    parser.add_argument("--split", type=str, default="s",
                        help="Dataset split to write")
    args = parser.parse_args()

    main(args)