// Copyright 2020 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "generator.h"

#include "definitions.h"
#include "instruction.pb.h"
#include "instruction.h"
#include "random_generator.h"
#include "absl/memory/memory.h"

namespace automl_zero
{

        using ::absl::make_unique;
        using ::std::endl;
        using ::std::make_shared;
        using ::std::mt19937;
        using ::std::shared_ptr;
        using ::std::vector;

        void PadComponentFunctionWithInstruction(
            const size_t total_instructions,
            const shared_ptr<const Instruction> &instruction,
            vector<shared_ptr<const Instruction>> *component_function)
        {
                component_function->reserve(total_instructions);
                while (component_function->size() < total_instructions)
                {
                        component_function->emplace_back(instruction);
                }
        }

        Generator::Generator(
            const HardcodedAlgorithmID init_model,
            const IntegerT setup_size_init,
            const IntegerT predict_size_init,
            const IntegerT learn_size_init,
            const vector<Op> &allowed_setup_ops,
            const vector<Op> &allowed_predict_ops,
            const vector<Op> &allowed_learn_ops,
            mt19937 *bit_gen,
            RandomGenerator *rand_gen)
            : init_model_(init_model),
              setup_size_init_(setup_size_init),
              predict_size_init_(predict_size_init),
              learn_size_init_(learn_size_init),
              allowed_setup_ops_(allowed_setup_ops),
              allowed_predict_ops_(allowed_predict_ops),
              allowed_learn_ops_(allowed_learn_ops),
              rand_gen_(rand_gen),
              randomizer_(
                  allowed_setup_ops,
                  allowed_predict_ops,
                  allowed_learn_ops,
                  bit_gen,
                  rand_gen_),
              no_op_instruction_(make_shared<const Instruction>()) {}

        Algorithm Generator::TheInitModel()
        {
                return ModelByID(init_model_);
        }

        Algorithm Generator::ModelByID(const HardcodedAlgorithmID model)
        {
                switch (model)
                {
                case NO_OP_ALGORITHM:
                        return NoOp();
                case RANDOM_ALGORITHM:
                        return Random();
                case NEURAL_NET_ALGORITHM:
                        return NeuralNet(
                            kDefaultLearningRate, 0.1, 0.1);
                case INTEGRATION_TEST_DAMAGED_NEURAL_NET_ALGORITHM:
                {
                        Algorithm algorithm = NeuralNet(
                            kDefaultLearningRate, 0.1, 0.1);
                        // Delete the first two instructions in setup which are the
                        // gaussian initialization of the first and final layer weights.
                        algorithm.setup_.erase(algorithm.setup_.begin());
                        algorithm.setup_.erase(algorithm.setup_.begin());
                        return algorithm;
                }
                case LINEAR_ALGORITHM:
                        return LinearModel(kDefaultLearningRate);
                default:
                        LOG(FATAL) << "Unsupported algorithm ID." << endl;
                }
        }

        inline void FillComponentFunctionWithInstruction(
            const IntegerT num_instructions,
            const shared_ptr<const Instruction> &instruction,
            vector<shared_ptr<const Instruction>> *component_function)
        {
                component_function->reserve(num_instructions);
                component_function->clear();
                for (IntegerT pos = 0; pos < num_instructions; ++pos)
                {
                        component_function->emplace_back(instruction);
                }
        }

        Algorithm Generator::NoOp()
        {
                Algorithm algorithm;
                FillComponentFunctionWithInstruction(
                    setup_size_init_, no_op_instruction_, &algorithm.setup_);
                FillComponentFunctionWithInstruction(
                    predict_size_init_, no_op_instruction_, &algorithm.predict_);
                FillComponentFunctionWithInstruction(
                    learn_size_init_, no_op_instruction_, &algorithm.learn_);
                return algorithm;
        }

        Algorithm Generator::Random()
        {
                Algorithm algorithm = NoOp();
                CHECK(setup_size_init_ == 0 || !allowed_setup_ops_.empty());
                CHECK(predict_size_init_ == 0 || !allowed_predict_ops_.empty());
                CHECK(learn_size_init_ == 0 || !allowed_learn_ops_.empty());
                randomizer_.Randomize(&algorithm);
                return algorithm;
        }

        void PadComponentFunctionWithRandomInstruction(
            const size_t total_instructions, const Op op,
            RandomGenerator *rand_gen,
            vector<shared_ptr<const Instruction>> *component_function)
        {
                component_function->reserve(total_instructions);
                while (component_function->size() < total_instructions)
                {
                        component_function->push_back(make_shared<const Instruction>(op, rand_gen));
                }
        }

        Generator::Generator()
            : init_model_(RANDOM_ALGORITHM),
              setup_size_init_(6),
              predict_size_init_(3),
              learn_size_init_(9),
              allowed_setup_ops_(
                  {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}),
              allowed_predict_ops_(
                  {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}),
              allowed_learn_ops_(
                  {NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP}),
              bit_gen_owned_(make_unique<mt19937>(GenerateRandomSeed())),
              rand_gen_owned_(make_unique<RandomGenerator>(bit_gen_owned_.get())),
              rand_gen_(rand_gen_owned_.get()),
              randomizer_(
                  allowed_setup_ops_,
                  allowed_predict_ops_,
                  allowed_learn_ops_,
                  bit_gen_owned_.get(),
                  rand_gen_),
              no_op_instruction_(make_shared<const Instruction>()) {}

        Algorithm Generator::UnitTestNeuralNetNoBiasNoGradient(
            const double learning_rate)
        {
                Algorithm algorithm;

                // Scalar addresses
                constexpr AddressT kLearningRateAddress = 2;
                constexpr AddressT kPredictionErrorAddress = 3;
                CHECK_GE(k_MAX_SCALAR_ADDRESSES, 4);

                // Vector addresses.
                constexpr AddressT kFinalLayerWeightsAddress = 3;
                CHECK_EQ(
                    kFinalLayerWeightsAddress,
                    Generator::kUnitTestNeuralNetNoBiasNoGradientFinalLayerWeightsAddress);
                constexpr AddressT kOneFollowedByZeroesVectorAddress = 4;
                CHECK_EQ(
                    kOneFollowedByZeroesVectorAddress,
                    Generator::kOneFollowedByZeroesVectorAddress);
                constexpr AddressT kFirstLayerOutputBeforeReluAddress = 5;
                constexpr AddressT kFirstLayerOutputAfterReluAddress = 6;
                constexpr AddressT kZerosAddress = 7;
                constexpr AddressT kGradientWrtFinalLayerWeightsAddress = 8;
                constexpr AddressT kGradientWrtActivationsAddress = 9;
                constexpr AddressT kGradientOfReluAddress = 10;
                CHECK_GE(k_MAX_VECTOR_ADDRESSES, 11);

                // Matrix addresses.
                constexpr AddressT kFirstLayerWeightsAddress = 1;
                CHECK_EQ(
                    kFirstLayerWeightsAddress,
                    Generator::kUnitTestNeuralNetNoBiasNoGradientFirstLayerWeightsAddress);
                constexpr AddressT kGradientWrtFirstLayerWeightsAddress = 2;
                CHECK_GE(k_MAX_MATRIX_ADDRESSES, 2);

                shared_ptr<const Instruction> no_op_instruction =
                    make_shared<const Instruction>();

                algorithm.setup_.emplace_back(make_shared<const Instruction>(
                    SCALAR_CONST_SET_OP,
                    kLearningRateAddress,
                    ActivationDataSetter(learning_rate)));
                // memory.vector_[Generator::kOneFollowedByZeroesVectorAddress](0) = 1;
                algorithm.setup_.emplace_back(make_shared<const Instruction>(
                    VECTOR_CONST_SET_OP,
                    kOneFollowedByZeroesVectorAddress,
                    FloatDataSetter(0),
                    FloatDataSetter(1)));
                PadComponentFunctionWithInstruction(
                    setup_size_init_, no_op_instruction, &algorithm.setup_);

                IntegerT num_predict_instructions = 5;
                algorithm.predict_.reserve(num_predict_instructions);
                // Multiply with first layer weight matrix.
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    MATRIX_VECTOR_PRODUCT_OP,
                    kFirstLayerWeightsAddress, k_FEATURES_VECTOR_ADDRESS,
                    kFirstLayerOutputBeforeReluAddress));
                // Apply RELU.
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    VECTOR_MAX_OP, kFirstLayerOutputBeforeReluAddress, kZerosAddress,
                    kFirstLayerOutputAfterReluAddress));
                // Dot product with final layer weight vector.
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    VECTOR_INNER_PRODUCT_OP, kFirstLayerOutputAfterReluAddress,
                    kFinalLayerWeightsAddress, k_PREDICTIONS_SCALAR_ADDRESS));
                // memory->vector_[k_PREDICTIONS_VECTOR_ADDRESS] = memory->scalar_[k_PREDICTIONS_SCALAR_ADDRESS] * {1, 0, 0, ...}
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_PRODUCT_OP,
                    k_PREDICTIONS_SCALAR_ADDRESS,
                    kOneFollowedByZeroesVectorAddress,
                    k_PREDICTIONS_VECTOR_ADDRESS));
                PadComponentFunctionWithInstruction(
                    predict_size_init_, no_op_instruction, &algorithm.predict_);

                algorithm.learn_.reserve(11);
                // memory->scalar_[k_PREDICTIONS_SCALAR_ADDRESS] = memory->vector_[k_PREDICTIONS_VECTOR_ADDRESS][0]
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_AT_INDEX_SET_OP,
                    k_PREDICTIONS_SCALAR_ADDRESS,
                    k_PREDICTIONS_VECTOR_ADDRESS,
                    FloatDataSetter(0)));
                // memory->scalar_[k_LABELS_SCALAR_ADDRESS] = memory->vector_[k_LABELS_VECTOR_ADDRESS][0]
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_AT_INDEX_SET_OP,
                    k_LABELS_SCALAR_ADDRESS,
                    k_LABELS_VECTOR_ADDRESS,
                    FloatDataSetter(0)));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_DIFF_OP, k_LABELS_SCALAR_ADDRESS, k_PREDICTIONS_SCALAR_ADDRESS,
                    kPredictionErrorAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_PRODUCT_OP,
                    kLearningRateAddress, kPredictionErrorAddress, kPredictionErrorAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_PRODUCT_OP, kPredictionErrorAddress,
                    kFirstLayerOutputAfterReluAddress, kGradientWrtFinalLayerWeightsAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_SUM_OP,
                    kFinalLayerWeightsAddress, kGradientWrtFinalLayerWeightsAddress,
                    kFinalLayerWeightsAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_PRODUCT_OP,
                    kPredictionErrorAddress, kFinalLayerWeightsAddress,
                    kGradientWrtActivationsAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_HEAVYSIDE_OP,
                    kFirstLayerOutputBeforeReluAddress, 0, kGradientOfReluAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_PRODUCT_OP,
                    kGradientOfReluAddress, kGradientWrtActivationsAddress,
                    kGradientWrtActivationsAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_OUTER_PRODUCT_OP,
                    kGradientWrtActivationsAddress, k_FEATURES_VECTOR_ADDRESS,
                    kGradientWrtFirstLayerWeightsAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    MATRIX_SUM_OP,
                    kFirstLayerWeightsAddress, kGradientWrtFirstLayerWeightsAddress,
                    kFirstLayerWeightsAddress));
                PadComponentFunctionWithInstruction(
                    learn_size_init_, no_op_instruction, &algorithm.learn_);

                return algorithm;
        }

        Algorithm Generator::NeuralNet(
            const double learning_rate,
            const double first_init_scale,
            const double final_init_scale)
        {
                Algorithm algorithm;

                // Scalar addresses
                constexpr AddressT kFinalLayerBiasAddress = 2;
                constexpr AddressT kLearningRateAddress = 3;
                constexpr AddressT kPredictionErrorAddress = 4;
                CHECK_GE(k_MAX_SCALAR_ADDRESSES, 5);

                // Vector addresses.
                constexpr AddressT kFirstLayerBiasAddress = 3;
                constexpr AddressT kOneFollowedByZeroesVectorAddress = 4;
                CHECK_EQ(
                    kOneFollowedByZeroesVectorAddress,
                    Generator::kOneFollowedByZeroesVectorAddress);
                constexpr AddressT kFinalLayerWeightsAddress = 5;
                constexpr AddressT kFirstLayerOutputBeforeReluAddress = 6;
                constexpr AddressT kFirstLayerOutputAfterReluAddress = 7;
                constexpr AddressT kZerosAddress = 8;
                constexpr AddressT kGradientWrtFinalLayerWeightsAddress = 9;
                constexpr AddressT kGradientWrtActivationsAddress = 10;
                constexpr AddressT kGradientOfReluAddress = 11;
                CHECK_GE(k_MAX_VECTOR_ADDRESSES, 12);

                // Matrix addresses.
                constexpr AddressT kFirstLayerWeightsAddress = 0;
                constexpr AddressT kGradientWrtFirstLayerWeightsAddress = 1;
                CHECK_GE(k_MAX_MATRIX_ADDRESSES, 2);

                shared_ptr<const Instruction> no_op_instruction =
                    make_shared<const Instruction>();

                algorithm.setup_.emplace_back(make_shared<const Instruction>(
                    VECTOR_GAUSSIAN_SET_OP,
                    kFinalLayerWeightsAddress,
                    FloatDataSetter(0.0),
                    FloatDataSetter(final_init_scale)));
                algorithm.setup_.emplace_back(make_shared<const Instruction>(
                    MATRIX_GAUSSIAN_SET_OP,
                    kFirstLayerWeightsAddress,
                    FloatDataSetter(0.0),
                    FloatDataSetter(first_init_scale)));
                algorithm.setup_.emplace_back(make_shared<const Instruction>(
                    SCALAR_CONST_SET_OP,
                    kLearningRateAddress,
                    ActivationDataSetter(learning_rate)));
                // memory.vector_[Generator::kOneFollowedByZeroesVectorAddress](0) = 1;
                algorithm.setup_.emplace_back(make_shared<const Instruction>(
                    VECTOR_CONST_SET_OP,
                    kOneFollowedByZeroesVectorAddress,
                    FloatDataSetter(0),
                    FloatDataSetter(1)));
                PadComponentFunctionWithInstruction(
                    setup_size_init_, no_op_instruction, &algorithm.setup_);

                // Multiply with first layer weight matrix.
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    MATRIX_VECTOR_PRODUCT_OP,
                    kFirstLayerWeightsAddress,
                    k_FEATURES_VECTOR_ADDRESS,
                    kFirstLayerOutputBeforeReluAddress));
                // Add first layer bias.
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    VECTOR_SUM_OP,
                    kFirstLayerOutputBeforeReluAddress,
                    kFirstLayerBiasAddress,
                    kFirstLayerOutputBeforeReluAddress));
                // Apply RELU.
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    VECTOR_MAX_OP,
                    kFirstLayerOutputBeforeReluAddress,
                    kZerosAddress,
                    kFirstLayerOutputAfterReluAddress));
                // Dot product with final layer weight vector.
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    VECTOR_INNER_PRODUCT_OP,
                    kFirstLayerOutputAfterReluAddress,
                    kFinalLayerWeightsAddress,
                    k_PREDICTIONS_SCALAR_ADDRESS));
                // Add final layer bias.
                CHECK_LE(kFinalLayerBiasAddress, k_MAX_SCALAR_ADDRESSES);
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    SCALAR_SUM_OP,
                    k_PREDICTIONS_SCALAR_ADDRESS,
                    kFinalLayerBiasAddress,
                    k_PREDICTIONS_SCALAR_ADDRESS));
                // memory->vector_[k_PREDICTIONS_VECTOR_ADDRESS] = memory->scalar_[k_PREDICTIONS_SCALAR_ADDRESS] * {1, 0, 0, ...}
                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_PRODUCT_OP,
                    k_PREDICTIONS_SCALAR_ADDRESS,
                    kOneFollowedByZeroesVectorAddress,
                    k_PREDICTIONS_VECTOR_ADDRESS));
                PadComponentFunctionWithInstruction(
                    predict_size_init_, no_op_instruction, &algorithm.predict_);

                algorithm.learn_.reserve(11);
                // memory->scalar_[k_PREDICTIONS_SCALAR_ADDRESS] = memory->vector_[k_PREDICTIONS_VECTOR_ADDRESS][0]
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_AT_INDEX_SET_OP,
                    k_PREDICTIONS_SCALAR_ADDRESS,
                    k_PREDICTIONS_VECTOR_ADDRESS,
                    FloatDataSetter(0)));
                // memory->scalar_[k_LABELS_SCALAR_ADDRESS] = memory->vector_[k_LABELS_VECTOR_ADDRESS][0]
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_AT_INDEX_SET_OP,
                    k_LABELS_SCALAR_ADDRESS,
                    k_LABELS_VECTOR_ADDRESS,
                    FloatDataSetter(0)));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_DIFF_OP,
                    k_LABELS_SCALAR_ADDRESS,
                    k_PREDICTIONS_SCALAR_ADDRESS,
                    kPredictionErrorAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_PRODUCT_OP,
                    kLearningRateAddress,
                    kPredictionErrorAddress,
                    kPredictionErrorAddress));
                CHECK_LE(kFinalLayerBiasAddress, k_MAX_SCALAR_ADDRESSES);
                // Update final layer bias.
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_SUM_OP,
                    kFinalLayerBiasAddress,
                    kPredictionErrorAddress,
                    kFinalLayerBiasAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_PRODUCT_OP,
                    kPredictionErrorAddress,
                    kFirstLayerOutputAfterReluAddress,
                    kGradientWrtFinalLayerWeightsAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_SUM_OP,
                    kFinalLayerWeightsAddress,
                    kGradientWrtFinalLayerWeightsAddress,
                    kFinalLayerWeightsAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_PRODUCT_OP,
                    kPredictionErrorAddress,
                    kFinalLayerWeightsAddress,
                    kGradientWrtActivationsAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_HEAVYSIDE_OP,
                    kFirstLayerOutputBeforeReluAddress,
                    0,
                    kGradientOfReluAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_PRODUCT_OP,
                    kGradientOfReluAddress,
                    kGradientWrtActivationsAddress,
                    kGradientWrtActivationsAddress));
                // Update first layer bias.
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_SUM_OP,
                    kFirstLayerBiasAddress,
                    kGradientWrtActivationsAddress,
                    kFirstLayerBiasAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_OUTER_PRODUCT_OP,
                    kGradientWrtActivationsAddress,
                    k_FEATURES_VECTOR_ADDRESS,
                    kGradientWrtFirstLayerWeightsAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    MATRIX_SUM_OP,
                    kFirstLayerWeightsAddress,
                    kGradientWrtFirstLayerWeightsAddress,
                    kFirstLayerWeightsAddress));
                PadComponentFunctionWithInstruction(
                    learn_size_init_, no_op_instruction, &algorithm.learn_);

                return algorithm;
        }

        Algorithm Generator::LinearModel(const double learning_rate)
        {
                Algorithm algorithm;

                // Scalar addresses
                constexpr AddressT kLearningRateAddress = 2;
                constexpr AddressT kPredictionErrorAddress = 3;
                CHECK_GE(k_MAX_SCALAR_ADDRESSES, 4);

                // Vector addresses.
                constexpr AddressT kWeightsAddress = 3;
                constexpr AddressT kCorrectionAddress = 4;
                CHECK_GE(k_MAX_VECTOR_ADDRESSES, 5);

                CHECK_GE(k_MAX_MATRIX_ADDRESSES, 0);

                shared_ptr<const Instruction> no_op_instruction =
                    make_shared<const Instruction>();

                algorithm.setup_.emplace_back(make_shared<const Instruction>(
                    SCALAR_CONST_SET_OP,
                    kLearningRateAddress,
                    ActivationDataSetter(learning_rate)));
                PadComponentFunctionWithInstruction(
                    setup_size_init_, no_op_instruction, &algorithm.setup_);

                algorithm.predict_.emplace_back(make_shared<const Instruction>(
                    VECTOR_INNER_PRODUCT_OP,
                    kWeightsAddress, k_FEATURES_VECTOR_ADDRESS, k_PREDICTIONS_SCALAR_ADDRESS));
                PadComponentFunctionWithInstruction(
                    predict_size_init_, no_op_instruction, &algorithm.predict_);

                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_DIFF_OP,
                    k_LABELS_SCALAR_ADDRESS, k_PREDICTIONS_SCALAR_ADDRESS,
                    kPredictionErrorAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_PRODUCT_OP,
                    kLearningRateAddress, kPredictionErrorAddress,
                    kPredictionErrorAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    SCALAR_VECTOR_PRODUCT_OP,
                    kPredictionErrorAddress, k_FEATURES_VECTOR_ADDRESS, kCorrectionAddress));
                algorithm.learn_.emplace_back(make_shared<const Instruction>(
                    VECTOR_SUM_OP,
                    kWeightsAddress, kCorrectionAddress, kWeightsAddress));
                PadComponentFunctionWithInstruction(
                    learn_size_init_, no_op_instruction, &algorithm.learn_);
                return algorithm;
        }

        Algorithm Generator::SortAlgorithm(const int F)
        {
                Algorithm algorithm;

                // Scalar addresses
                CHECK_GE(k_MAX_SCALAR_ADDRESSES, 3);

                // Vector addresses.
                CHECK_GE(k_MAX_VECTOR_ADDRESSES, 5);

                shared_ptr<const Instruction> no_op_instruction = make_shared<const Instruction>();

                SortAlgorithmSetup(algorithm.setup_, no_op_instruction);
                SortAlgorithmPredict(algorithm.predict_, F, no_op_instruction);
                SortAlgorithmLearn(algorithm.learn_, no_op_instruction);

                return algorithm;
        }

        void Generator::SortAlgorithmSetup(std::vector<std::shared_ptr<const Instruction>> &setup, const std::shared_ptr<const Instruction> &no_op_instruction)
        {
                PadComponentFunctionWithInstruction(setup_size_init_, no_op_instruction, &setup);
        }

        void Generator::SortAlgorithmPredict(std::vector<std::shared_ptr<const Instruction>> &predict, const int F, const std::shared_ptr<const Instruction> &no_op_instruction)
        {
                constexpr AddressT kConstOneAddress = 2;
                CHECK_EQ(
                    kConstOneAddress,
                    Generator::kConstOneAddress);

                // s2 = 1
                predict.emplace_back(std::make_shared<const Instruction>(
                    SCALAR_CONST_SET_OP,
                    kConstOneAddress,
                    ActivationDataSetter(1.0)));
                // s3 = 0.5
                predict.emplace_back(std::make_shared<const Instruction>(
                    SCALAR_CONST_SET_OP,
                    3,
                    ActivationDataSetter(0.5)));
                // v1 = bcast(s2)
                predict.emplace_back(std::make_shared<const Instruction>(
                    SCALAR_BROADCAST_OP,
                    kConstOneAddress,
                    1));
                // s4 = dot(v1, v1)
                predict.emplace_back(std::make_shared<const Instruction>(
                    VECTOR_INNER_PRODUCT_OP,
                    1,
                    1,
                    4));
                // s5 =  s4 - s2
                predict.emplace_back(std::make_shared<const Instruction>(
                    SCALAR_DIFF_OP,
                    4,
                    2,
                    5));
                // v_k_PREDICTIONS_VECTOR_ADDRESS = s2 * v_k_FEATURES_VECTOR_ADDRESS
                predict.emplace_back(std::make_shared<const Instruction>(
                    SCALAR_VECTOR_PRODUCT_OP,
                    kConstOneAddress,
                    k_FEATURES_VECTOR_ADDRESS,
                    k_PREDICTIONS_VECTOR_ADDRESS));
                // for(s6 = 1..s5) {
                Instruction loop = Instruction(LOOP, 5, 6);
                //     s0 = s6 - s2
                loop.children_.emplace_back(std::make_shared<const Instruction>(
                    SCALAR_DIFF_OP,
                    6,
                    2,
                    0));
                //     s0 = s0 + s3
                loop.children_.emplace_back(std::make_shared<const Instruction>(
                    SCALAR_SUM_OP,
                    0,
                    3,
                    0));
                //     s0 = s0 / s4
                loop.children_.emplace_back(std::make_shared<const Instruction>(
                    SCALAR_DIVISION_OP,
                    0,
                    4,
                    0));
                //     s1 = arg_min(v_k_PREDICTIONS_VECTOR_ADDRESS, s0)
                loop.children_.emplace_back(std::make_shared<const Instruction>(
                    VECTOR_ARG_MIN_OP,
                    k_PREDICTIONS_VECTOR_ADDRESS,
                    0,
                    1));
                //     swap(v_k_PREDICTIONS_VECTOR_ADDRESS, s0, s1)
                loop.children_.emplace_back(std::make_shared<const Instruction>(
                    VECTOR_SWAP_OP,
                    0,
                    1,
                    k_PREDICTIONS_VECTOR_ADDRESS));
                // }
                predict.emplace_back(std::make_shared<const Instruction>(loop));

                PadComponentFunctionWithInstruction(predict_size_init_, no_op_instruction, &predict);
        }

        void Generator::SortAlgorithmLearn(std::vector<std::shared_ptr<const Instruction>> &learn, const std::shared_ptr<const Instruction> &no_op_instruction)
        {
                PadComponentFunctionWithInstruction(learn_size_init_, no_op_instruction, &learn);
        }
} // namespace automl_zero
