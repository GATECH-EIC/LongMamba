# longbench-e dataset
CUDA_VISIBLE_DEVICES=0 python my_evaluation.py --model_arch vanilla -lt e --model state-spaces/mamba2-1.3b &
CUDA_VISIBLE_DEVICES=1 python my_evaluation.py --model_arch ours -lt e --align_path ablation-clampTop0.05max5-1 --our_method dt_thre --model state-spaces/mamba2-1.3b &

# PG19 dataset
CUDA_VISIBLE_DEVICES=3 python my_evaluation.py --model_arch vanilla -dt pg19 --model state-spaces/mamba2-1.3b &
CUDA_VISIBLE_DEVICES=4 python my_evaluation.py --model_arch ours -dt pg19 --align_path ablation-clampTop0.05max5-1 --our_method dt_thre --model state-spaces/mamba2-1.3b &

# your customized dataset (.txt) (specify the file path with --sample_path)
CUDA_VISIBLE_DEVICES=5 python my_evaluation.py --model_arch vanilla -ppl --model state-spaces/mamba2-1.3b --sample_path subseq_lambada.txt &
CUDA_VISIBLE_DEVICES=6 python my_evaluation.py --model_arch ours -ppl --align_path ablation-clampTop0.05max5-1 --our_method dt_thre --model state-spaces/mamba2-1.3b --sample_path subseq_lambada.txt &
