
# # longbench
# python my_evaluation.py -d 0 --model_arch ours -lt yes --align_path thepileavg --our_method bound --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 1 --model_arch ours -lt yes --align_path thepileavg --our_method alpha --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 2 --model_arch ours -lt yes --align_path thepileavg --our_method offline --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 3 --model_arch deci -lt yes --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 4 --model_arch vanilla -lt yes --model state-spaces/mamba-2.8b &
# wait

# # l-eval Open-ended
# python my_evaluation.py -d 0 --model_arch ours -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --align_path thepileavg --our_method bound --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 1 --model_arch ours -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --align_path thepileavg --our_method alpha --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 2 --model_arch ours -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --align_path thepileavg --our_method offline --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 3 --model_arch deci -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 4 --model_arch vanilla -le ngram_eval --Leval_task LEval-data/Open-ended-tasks --model state-spaces/mamba-2.8b &
# wait

# # perplexity subseq_thepile.txt
# python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_thepile.txt --align_path thepileavg --our_method bound --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_thepile.txt --align_path thepileavg --our_method alpha --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_thepile.txt --align_path thepileavg --our_method offline --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 3 --model_arch deci -ppl --sample_path subseq_thepile.txt --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 4 --model_arch vanilla -ppl --sample_path subseq_thepile.txt --model state-spaces/mamba-2.8b &
# wait

# # perplexity subseq_lambada.txt
# python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepileavg --our_method bound --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepileavg --our_method alpha --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepileavg --our_method offline --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 3 --model_arch deci -ppl --sample_path subseq_lambada.txt --model state-spaces/mamba-2.8b &
# python my_evaluation.py -d 4 --model_arch vanilla -ppl --sample_path subseq_lambada.txt --model state-spaces/mamba-2.8b &
# wait
# python my_evaluation.py -d 0 --model_arch vanilla -lt yes --model state-spaces/mamba2-130m &
# python my_evaluation.py -d 2 --model_arch vanilla -lt yes --model state-spaces/mamba2-780m &
# python my_evaluation.py -d 3 --model_arch vanilla -lt yes --model state-spaces/mamba2-1.3m &
python my_evaluation.py -d 0 --model_arch vanilla -dt pg19 --model state-spaces/mamba2-1.3b &
python my_evaluation.py -d 2 --model_arch vanilla -lt yes --model state-spaces/mamba2-780m &
python my_evaluation.py -d 3 --model_arch vanilla -lt yes --model state-spaces/mamba2-1.3b &
# python my_evaluation.py -d 4 --model_arch ours -dt pg19 --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-8 &
# python my_evaluation.py -d 5 --model_arch ours -lt yes --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-8 &
# python my_evaluation.py -d 6 --model_arch ours -lt yes --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-8 &
python my_evaluation.py -d 4 --model_arch vanilla -dt pg19 --model state-spaces/mamba2-130m &
python my_evaluation.py -d 5 --model_arch vanilla -dt pg19 --model state-spaces/mamba2-780m &
python my_evaluation.py -d 6 --model_arch vanilla -lt yes --model state-spaces/mamba2-130m &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-31 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-130m --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-31 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-31 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-2 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-130m --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-2 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-780m --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-2 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method bound --model state-spaces/mamba2-1.3b --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-2 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-2 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-2 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-2 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-1.3b --c 1e-3 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-5 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-7 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-9 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-11 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-13 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-15 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-17 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-4 &
wait
python my_evaluation.py -d 0 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-19 &
python my_evaluation.py -d 2 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-21 &
python my_evaluation.py -d 3 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-23 &
python my_evaluation.py -d 4 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-25 &
python my_evaluation.py -d 5 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-27 &
python my_evaluation.py -d 6 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-29 &
python my_evaluation.py -d 7 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-2 &
python my_evaluation.py -d 1 --model_arch ours -ppl --sample_path subseq_lambada.txt --align_path thepile_newavg --our_method dt_thre --model state-spaces/mamba2-780m --c 1e-3 &
wait


# python my_evaluation.py -d 0 --model_arch vanilla -dt pg19 --model state-spaces/mamba2-1.3b &
# python my_evaluation.py -d 2 --model_arch vanilla -lt yes --model state-spaces/mamba2-780m &
# python my_evaluation.py -d 3 --model_arch vanilla -lt yes --model state-spaces/mamba2-1.3b &
# python my_evaluation.py -d 4 --model_arch ours -dt pg19 --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-8 &
# python my_evaluation.py -d 5 --model_arch ours -lt yes --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-8 &
# python my_evaluation.py -d 6 --model_arch ours -lt yes --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-8 &
# wait
# python my_evaluation.py -d 0 --model_arch vanilla -dt pg19 --model state-spaces/mamba2-130m &
# python my_evaluation.py -d 2 --model_arch vanilla -dt pg19 --model state-spaces/mamba2-780m &
# python my_evaluation.py -d 3 --model_arch vanilla -lt yes --model state-spaces/mamba2-130m &
# python my_evaluation.py -d 4 --model_arch ours -dt pg19 --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-8 &
# python my_evaluation.py -d 5 --model_arch ours -dt pg19 --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-8 &
# python my_evaluation.py -d 6 --model_arch ours -lt yes --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-130m --c 1e-8 &
# wait
python my_evaluation.py -d 3 --model_arch vanilla -lt yes --model state-spaces/mamba2-1.3b &
python my_evaluation.py -d 4 --model_arch ours -lt yes --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-2 &
python my_evaluation.py -d 5 --model_arch ours -lt yes --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-2 &
python my_evaluation.py -d 1 --model_arch ours -lt yes --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-2 &
python my_evaluation.py -d 2 --model_arch ours -lt yes --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-2 &

python my_evaluation.py -d 0 --model_arch ours -dt pg19 --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-1.3b --c 1e-2 && python my_evaluation.py -d 0 --model_arch ours -dt pg19 --align_path thepile_newavg --our_method norm --model state-spaces/mamba2-780m --c 1e-2 &
python my_evaluation.py -d 6 --model_arch ours -dt pg19 --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-1.3b --c 1e-2 && python my_evaluation.py -d 6 --model_arch ours -dt pg19 --align_path thepile_newavg --our_method alpha --model state-spaces/mamba2-780m --c 1e-2 &
python my_evaluation.py -d 7 --model_arch vanilla -dt pg19 --model state-spaces/mamba2-1.3b &
wait

