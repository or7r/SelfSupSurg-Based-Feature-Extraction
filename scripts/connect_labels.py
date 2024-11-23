import os
import pickle
import logging
import itertools

def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pkl(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def connect_labels(files_list):
    labels = {}
    for file in files_list:
        data = read_pkl(file)

        def fix_id(row):
            row = row.copy()
            vid_name = row["Frame_id"].split("_")[0]
            dataset_name = vid_name[:3]
            vid_id = int(vid_name[3:])
            frame_id = int(row["Frame_id"].split("_")[1])
            
            if "SBP" in vid_name:
                vid_id += 100 # to avoid overlap with other dataset
            
            row["unique_id"] = vid_id * 10 ** 8 + frame_id
            return row


        for vid_name in data.keys():
            data[vid_name] = [fix_id(x) for x in data[vid_name]]

        # assert that unique ids are unique and do not overlap with previous files
        a1 = [x["unique_id"] for x in itertools.chain(*list(labels.values()))]
        a2 = [x["unique_id"] for x in itertools.chain(*list(data.values()))]
        assert len(set(a1).intersection(set(a2))) == 0

        labels.update(data)
    return labels

if __name__ == '__main__':

    splits = ['train', 'val', 'test']

    for split in splits:

        file_name = f'1fps{"_100" if split == "train" else ""}_0.pickle'


        src_files = [f"datasets/MultiBypass140/raw_labels/bern/{split}/{file_name}",
                    f"datasets/MultiBypass140/raw_labels/strasbourg/{split}/{file_name}"]
        dst_file = f"datasets/MultiBypass140/labels/{split}/{file_name}"

        # make sure the destination directory exists
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

        labels = connect_labels(src_files)
        write_pkl(dst_file, labels)
