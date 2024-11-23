import os
import argparse
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--label_file', type=str, required=True)
    return parser.parse_args()


def get_file_list(label_file):
    with open(label_file, 'rb') as f:
        label = pickle.load(f)
    return list(label.keys())


def main():
    args = parse_args()
    file_list = get_file_list(args.label_file)

    os.makedirs(args.output_dir, exist_ok=True)

    for file in file_list:
        os.symlink(os.path.join(args.input_dir, file), 
                   os.path.join(args.output_dir, file))
        

if __name__ == '__main__':
    main()
