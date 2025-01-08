gpu_ids=(0 1 2 3 4 5 6 7)

# for clamp in 0.15 0.16 0.17 0.18 0.19
# do
#     for c in 1e-29 1e-30 1e-31 1e-32 1e-33 1e-34 1e-35
#     do
#         echo "thepile_new-clampTop$clamp"
#         b_values_sets=(
#             "0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4" # "1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1"
#         )

#         for b_values in "${b_values_sets[@]}"
#         do
#             IFS=' ' read -r -a b_array <<< "$b_values"

#             for i in "${!gpu_ids[@]}"
#             do
#                 if [ $i -lt ${#b_array[@]} ]; then
#                     b=${b_array[$i]}
#                     gpu_id=${gpu_ids[$i]}
#                     echo "Running: c=$c, b=$b on GPU $gpu_id"
#                     python my_evaluation.py \
#                     -d $gpu_id \
#                     --model_arch ours \
#                     -ppl \
#                     --sample_path subseq_lambada.txt \
#                     --align_path "thepile_new-clampTop$clamp-max" \
#                     --our_method dt_thre \
#                     --model state-spaces/mamba-1.4b \
#                     --c $c \
#                     --b $b &
#                 fi
#             done
#             wait
#         done
#     done
# done

for clamp in 0.15 0.16 0.17 0.18 0.19
do
    for b in 1.0
    do
        echo "thepile_new-clampTop$clamp"
        c_values_sets=(
            "1e-29 1e-30 1e-31 1e-32 1e-33 1e-34 1e-35" # "1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1"
            "1e-26 1e-27 1e-28 1e-36 1e-37 1e-38 1e-39"
            "1e-40 1e-20 1e-21 1e-22 1e-23 1e-24 1e-25"
        )

        for c_values in "${c_values_sets[@]}"
        do
            IFS=' ' read -r -a c_array <<< "$c_values"

            for i in "${!gpu_ids[@]}"
            do
                if [ $i -lt ${#c_array[@]} ]; then
                    c=${c_array[$i]}
                    gpu_id=${gpu_ids[$i]}
                    echo "Running: c=$c, b=$b on GPU $gpu_id"
                    python my_evaluation.py \
                    -d $gpu_id \
                    --model_arch ours \
                    -ppl \
                    --sample_path subseq_lambada.txt \
                    --align_path "thepile_new-clampTop$clamp-max" \
                    --our_method dt_thre \
                    --model state-spaces/mamba2-1.3b \
                    --c $c \
                    --b $b &
                fi
            done
            wait
        done
    done
done
