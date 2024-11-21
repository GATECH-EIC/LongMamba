import random
import os

num_codes = 10000
codes = []
save_path = './tmp'

for i in range(num_codes):
    cur_code = str(random.randint(0,99999))
    codes.append(cur_code)
    # zero_pad = 5-len(cur_code)
    # codes.append('0'*zero_pad + cur_code)


with open(os.path.join(save_path,'codes.txt'), 'w') as f:
    f.write('\n'.join(codes))

f.close()




