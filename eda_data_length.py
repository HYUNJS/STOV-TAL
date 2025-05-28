import json, os
import os.path as osp
import pandas as pd


TH_K400_filepath = "data/thumos14/annotations/validation_K400_tal.json"
TH_nonK400_filepath = "data/thumos14/annotations/validation_nonK400_tal.json"
FA_K400_filepath = "data/fineaction/annotations/validation_K400_tal.json"
FA_nonK400_filepath = "data/fineaction/annotations/validation_nonK400_tal.json"

with open(TH_K400_filepath, "r") as fp:
    TH_K400_annos = json.load(fp)['database']
with open(FA_K400_filepath, "r") as fp:
    FA_K400_annos = json.load(fp)['database']
with open(TH_nonK400_filepath, "r") as fp:
    TH_nonK400_annos = json.load(fp)['database']
with open(FA_nonK400_filepath, "r") as fp:
    FA_nonK400_annos = json.load(fp)['database']

def get_segm_duration(annos):
    segm_list = []
    for vid in annos.keys():
        df = pd.DataFrame(annos[vid]['annotations'])['segment']
        segm_list.append(df)
    segms = pd.concat(segm_list).reset_index(drop=True)
    segm_duration = segms.apply(lambda x: x[1] - x[0]).values
    return segm_duration

def get_stat_duration(duration):
    print(f"Avg. {duration.mean():.2f} | Std. {duration.std():.2f} | Min. {duration.min()} | Max. {duration.max()}")

TH_K400_segm_duration = get_segm_duration(TH_K400_annos)
FA_K400_segm_duration = get_segm_duration(FA_K400_annos)
TH_nonK400_segm_duration = get_segm_duration(TH_nonK400_annos)
FA_nonK400_segm_duration = get_segm_duration(FA_nonK400_annos)

print("duration")
print("TH")
get_stat_duration(TH_K400_segm_duration)
get_stat_duration(TH_nonK400_segm_duration)
print("FA")
get_stat_duration(FA_K400_segm_duration)
get_stat_duration(FA_nonK400_segm_duration)
print()
print("#instance")
print("TH")
print(len(TH_K400_segm_duration) / len(TH_K400_annos))
print(len(TH_nonK400_segm_duration) / len(TH_nonK400_annos))
print("FA")
print(len(FA_K400_segm_duration) / len(FA_K400_annos))
print(len(FA_nonK400_segm_duration) / len(FA_nonK400_annos))
