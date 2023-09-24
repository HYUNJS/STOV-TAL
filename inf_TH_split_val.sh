CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14/vifi_prompt_K400_softmax.yaml
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid.yaml -e 5&
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid.yaml -e 10&

# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid.yaml -e 35&
# CUDA_VISIBLE_DEVICES=0 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid.yaml -e 30&
# CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid.yaml -e 25&
# CUDA_VISIBLE_DEVICES=1 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid.yaml -e 20&

# CUDA_VISIBLE_DEVICES=2 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid_pseudo_0-05.yaml -e 35&
# CUDA_VISIBLE_DEVICES=2 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid_pseudo_0-05.yaml -e 30&
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid_pseudo_0-05.yaml -e 25&
# CUDA_VISIBLE_DEVICES=3 python inference.py configs/thumos14/vifi_prompt_K400_sigmoid_pseudo_0-05.yaml -e 20&
