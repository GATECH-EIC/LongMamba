
# longbench
python my_evaluation.py -d 0 --model_arch ours -lt yes --align_path thepileavg --our_method bound --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 1 --model_arch ours -lt yes --align_path thepileavg --our_method alpha --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 2 --model_arch ours -lt yes --align_path thepileavg --our_method offline --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 3 --model_arch deci -lt yes --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 4 --model_arch vanilla -lt yes --model state-spaces/mamba-2.8b &
wait

# l-eval Open-ended
python my_evaluation.py -d 0 --model_arch ours -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --align_path thepileavg --our_method bound --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 1 --model_arch ours -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --align_path thepileavg --our_method alpha --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 2 --model_arch ours -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --align_path thepileavg --our_method offline --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 3 --model_arch deci -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 4 --model_arch vanilla -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --model state-spaces/mamba-2.8b &
wait

# perplexity subseq_thepile.txt
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_thepile.txt --align_path thepileavg --our_method bound --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_thepile.txt --align_path thepileavg --our_method alpha --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_thepile.txt --align_path thepileavg --our_method offline --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 3 --model_arch deci -ppl --sample_path subseq_thepile.txt --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 4 --model_arch vanilla -ppl --sample_path subseq_thepile.txt --model state-spaces/mamba-2.8b &
wait

# perplexity subseq_lambada.txt
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepileavg --our_method bound --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepileavg --our_method alpha --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepileavg --our_method offline --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 3 --model_arch deci -ppl --sample_path subseq_lambada.txt --model state-spaces/mamba-2.8b &
python my_evaluation.py -d 4 --model_arch vanilla -ppl --sample_path subseq_lambada.txt --model state-spaces/mamba-2.8b &
wait