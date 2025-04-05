

import sys
dimension=int(sys.argv[1])
build_index_parameters = dict()
distance_metric = 'cosine'

index_parameters = {'R': '64', 'L': '128', 'M': '8', 'B': '0.45'}

B_value  = (0.25 + (250000*(4*int(index_parameters['R']) + 4*dimension)) / 2 ** 30)
index_parameters['B'] = str(B_value)[:4]

print("B value: ", index_parameters['B'])

import subprocess
target_train_fvecs = '<target_file_location>:/local/eric/ft-vect-train/ft_size_' + str(dimension) + '/' + 'ft_' + str(dimension) + '.pack'
target_validation_fvecs = '<target_file_location>:/local/eric/ft-vect-validation/ft_size_' + str(dimension) + '/' + 'ft_' + str(dimension) + '.pack'
train_fvecs_name = 'ft_train_' + str(dimension) + '.pack'
validation_fvecs_name = 'ft_validation_' + str(dimension) + '.pack'
result = subprocess.run(["rm", "-r", "data/tmp/"], capture_output=True, text=True)
result = subprocess.run(["mkdir", "data/tmp/"], capture_output=True, text=True)

result = subprocess.run(['scp', target_train_fvecs, 'data/tmp/'+train_fvecs_name], capture_output=True, text=True)
result = subprocess.run(['scp', target_validation_fvecs, 'data/tmp/'+validation_fvecs_name], capture_output=True, text=True)
result = subprocess.run(['ls', 'data/tmp/'], capture_output=True, text=True)

# convert the fvecs to fbin
target_index_fbin = "ft_index_" + str(dimension) + ".fbin"
target_query_fbins = "ft_query_" + str(dimension) + ".fbin"
result = subprocess.run(["./apps/utils/fvecs_to_bin", "float", "data/tmp/"+train_fvecs_name, "data/tmp/"+target_index_fbin], capture_output=True, text=True)
result = subprocess.run(["./apps/utils/fvecs_to_bin", "float", "data/tmp/"+validation_fvecs_name, "data/tmp/"+target_query_fbins], capture_output=True, text=True)
result = subprocess.run(['rm', 'data/tmp/'+train_fvecs_name], capture_output=True, text=True)
result = subprocess.run(['rm', 'data/tmp/'+validation_fvecs_name], capture_output=True, text=True)

# compute the ground truth
gt_file_name = "ft_gt_" + str(dimension) + "_gt1000"
result = subprocess.run(["./apps/utils/compute_groundtruth", "--data_type", "float", "--dist_fn", distance_metric, "--base_file", "data/tmp/"+target_index_fbin, "--query_file", "data/tmp/"+target_query_fbins, "--gt_file", "data/tmp/"+gt_file_name, "--K", "1000"], capture_output=True, text=True)
print(result)


graph_index = "ft_graph_" + str(dimension)
result = subprocess.run(['./apps/build_disk_index', '--data_type', 'float', '--dist_fn', distance_metric, '--data_path', 'data/tmp/' + target_index_fbin, '--index_path_prefix', 'data/tmp/' + graph_index, '-R', index_parameters['R'], '-L'+index_parameters['L'], '-B' , index_parameters['B'], '-M', index_parameters['M']], capture_output=True, text=True)
print(result.stdout)


result_path = "data/tmp/res"
result = subprocess.run(['./apps/search_disk_index', '--data_type', 'float', '--dist_fn', distance_metric, '--index_path_prefix', 'data/tmp/' + graph_index, '--query_file', 'data/tmp/' + target_query_fbins, '--gt_file', 'data/tmp/' + gt_file_name, '--result_path', result_path, '-K', '10', '-L', '10', '20', '30', '40', '50', '100', '--num_nodes_to_cache', '10000'], capture_output=True, text=True)
print(result.stdout)
#write stdout to file
with open('data/tmp/result.txt', 'w') as f:
    f.write(result.stdout)


subprocess.run(['mv', 'data/tmp/', 'data/MRL_' + str(dimension) + '_R' + index_parameters['R'] + '_L' + index_parameters['L'] + '_M' + index_parameters['M'] + '_B' + index_parameters['B'] + "_" + distance_metric])
