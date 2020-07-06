# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

# Usage:
# ./run_demo.sh

bazel run -c opt \
  --copt=-DMAX_SCALAR_ADDRESSES=4 \
  --copt=-DMAX_VECTOR_ADDRESSES=3 \
  --copt=-DMAX_MATRIX_ADDRESSES=1 \
  :run_search_experiment -- \
  --search_experiment_spec=" \
    search_tasks { \
      tasks { \
        unit_test_fixed_task { \
          train_features {elements: [0.41, 0.42, 0.43, 0.44]} \
          train_features {elements: [0.51, 0.52, 0.53, 0.54]} \
          train_features {elements: [0.61, 0.62, 0.63, 0.64]} \
          train_labels {elements: [4.1]} \
          train_labels {elements: [5.1]} \
          train_labels {elements: [6.1]} \
          valid_features {elements: [0.71, 0.72, 0.73, 0.74]} \
          valid_features {elements: [0.81, 0.82, 0.83, 0.84]} \
          valid_labels {elements: [7.1]} \
          valid_labels {elements: [8.1]} \
        } \
        features_size: 4 \
        num_train_examples: 3 \
        num_train_epochs: 8 \
        num_valid_examples: 2 \
        num_tasks: 1 \
        eval_type: RMS_ERROR \
      } \
    } \
    setup_ops: [SCALAR_CONST_SET_OP, VECTOR_INNER_PRODUCT_OP, SCALAR_DIFF_OP, SCALAR_PRODUCT_OP, SCALAR_VECTOR_PRODUCT_OP, VECTOR_SUM_OP] \
    predict_ops: [SCALAR_CONST_SET_OP, VECTOR_INNER_PRODUCT_OP, SCALAR_DIFF_OP, SCALAR_PRODUCT_OP, SCALAR_VECTOR_PRODUCT_OP, VECTOR_SUM_OP] \
    learn_ops: [SCALAR_CONST_SET_OP, VECTOR_INNER_PRODUCT_OP, SCALAR_DIFF_OP, SCALAR_PRODUCT_OP, SCALAR_VECTOR_PRODUCT_OP, VECTOR_SUM_OP] \
    learn_size_init: 8 \
    setup_size_init: 10 \
    predict_size_init: 2 \
    fec {num_train_examples: 3 num_valid_examples: 2} \
    fitness_combination_mode: MEAN_FITNESS_COMBINATION \
    population_size: 1000 \
    tournament_size: 10 \
    initial_population: RANDOM_ALGORITHM \
    max_train_steps: 2000000 \
    allowed_mutation_types {
      mutation_types: [ALTER_PARAM_MUTATION_TYPE, RANDOMIZE_INSTRUCTION_MUTATION_TYPE, RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE] \
    } \
    mutate_prob: 0.9 \
    progress_every: 10000 \
    " \
  --final_tasks=" \
    tasks { \
        unit_test_fixed_task { \
          train_features {elements: [0.41, 0.42, 0.43, 0.44]} \
          train_features {elements: [0.51, 0.52, 0.53, 0.54]} \
          train_features {elements: [0.61, 0.62, 0.63, 0.64]} \
          train_labels {elements: [4.1]} \
          train_labels {elements: [5.1]} \
          train_labels {elements: [6.1]} \
          valid_features {elements: [0.71, 0.72, 0.73, 0.74]} \
          valid_features {elements: [0.81, 0.82, 0.83, 0.84]} \
          valid_labels {elements: [7.1]} \
          valid_labels {elements: [8.1]} \
        } \
        features_size: 4 \
        num_train_examples: 3 \
        num_train_epochs: 8 \
        num_valid_examples: 2 \
        num_tasks: 1 \
        eval_type: RMS_ERROR \
      data_seeds: [1000000] \
      param_seeds: [2000000] \
    } \
    " \
  --max_experiments=0 \
  --randomize_task_seeds \
  --select_tasks=" \
    tasks { \
        unit_test_fixed_task { \
          train_features {elements: [0.41, 0.42, 0.43, 0.44]} \
          train_features {elements: [0.51, 0.52, 0.53, 0.54]} \
          train_features {elements: [0.61, 0.62, 0.63, 0.64]} \
          train_labels {elements: [4.1]} \
          train_labels {elements: [5.1]} \
          train_labels {elements: [6.1]} \
          valid_features {elements: [0.71, 0.72, 0.73, 0.74]} \
          valid_features {elements: [0.81, 0.82, 0.83, 0.84]} \
          valid_labels {elements: [7.1]} \
          valid_labels {elements: [8.1]} \
        } \
        features_size: 4 \
        num_train_examples: 3 \
        num_train_epochs: 8 \
        num_valid_examples: 2 \
        num_tasks: 1 \
        eval_type: RMS_ERROR \
    } \
    " \
  --sufficient_fitness=0.99
