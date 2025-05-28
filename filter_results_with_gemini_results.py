import os, json, argparse
import os.path as osp
import pandas as pd

'''
python filter_results_with_gemini_results.py --dataset_name fineaction --dataset_split con-novel
'''

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_name", choices=["thumos14", "fineaction"], required=True)
argparser.add_argument("--dataset_split", choices=["gen-all", "con-base", "con-novel"], required=True)
args = argparser.parse_args()

dataset_name = args.dataset_name
dataset_split = args.dataset_split

if dataset_name == "thumos14":
    dataset_shortname = "TH"
    final_epoch = "035"
elif dataset_name == "fineaction":
    dataset_shortname = "FA"
    final_epoch = "015"
    
if dataset_split == 'gen-all':
    split_name = ""
elif dataset_split == 'con-base':
    split_name = "_K400"
elif dataset_split == 'con-novel':
    split_name = "_nonK400"


stovtal_dir = f"ckpt/{dataset_shortname}_agn_K400/{dataset_name}_vifi_prop_K400_1/proposal_CLIP_cls_{dataset_shortname}_validation{split_name}_tal_ema/"
stovtal_filename = f"{dataset_name}_vifi_prop_K400_1_epoch_{final_epoch}.json"
new_stovtal_filename = stovtal_filename.replace(".json", "_gemini-vids.json")
gemini_dir = "gemini_output/fineaction/"
gemini_filename = f"gemini-1.5-flash-001_time-inst-v1_fineaction-{dataset_split}_results.json"

with open(osp.join(stovtal_dir, stovtal_filename), "r") as fp:
    stovtal_results = json.load(fp)['results']

with open(osp.join(gemini_dir, gemini_filename), "r") as fp:
    gemini_results = json.load(fp)['results']


gemini_vids = list(gemini_results.keys())

new_stovtal_results = {}
for vid in gemini_vids:
    new_stovtal_results[vid] = stovtal_results[vid]
    
print(f"[Save] {osp.join(stovtal_dir, new_stovtal_filename)}")
# with open(osp.join(stovtal_dir, new_stovtal_filename), "w") as fp:
#     json.dump({"results": new_stovtal_results}, fp)
