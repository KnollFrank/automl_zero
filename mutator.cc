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

#include "mutator.h"

#include <memory>
#include <vector>

#include "definitions.h"
#include "random_generator.h"
#include "absl/memory/memory.h"

namespace automl_zero
{

    using ::absl::make_unique; // NOLINT
    using ::std::endl;         // NOLINT
    using ::std::make_shared;  // NOLINT
    using ::std::mt19937;      // NOLINT
    using ::std::shared_ptr;   // NOLINT
    using ::std::vector;       // NOLINT

    Mutator::Mutator(
        const MutationTypeList &allowed_actions,
        const double mutate_prob,
        const vector<Op> &allowed_setup_ops,
        const vector<Op> &allowed_predict_ops,
        const vector<Op> &allowed_learn_ops,
        const IntegerT setup_size_min,
        const IntegerT setup_size_max,
        const IntegerT predict_size_min,
        const IntegerT predict_size_max,
        const IntegerT learn_size_min,
        const IntegerT learn_size_max,
        mt19937 *bit_gen,
        RandomGenerator *rand_gen)
        : allowed_actions_(allowed_actions),
        mutate_prob_(mutate_prob),
        allowed_setup_ops_(allowed_setup_ops),
        allowed_predict_ops_(allowed_predict_ops),
        allowed_learn_ops_(allowed_learn_ops),
        mutate_setup_(!allowed_setup_ops_.empty()),
        mutate_predict_(!allowed_predict_ops_.empty()),
        mutate_learn_(!allowed_learn_ops_.empty()),
        setup_size_min_(setup_size_min),
        setup_size_max_(setup_size_max),
        predict_size_min_(predict_size_min),
        predict_size_max_(predict_size_max),
        learn_size_min_(learn_size_min),
        learn_size_max_(learn_size_max),
        bit_gen_(bit_gen),
        rand_gen_(rand_gen),
        randomizer_(
            allowed_setup_ops_,
            allowed_predict_ops_,
            allowed_learn_ops_,
            bit_gen_,
            rand_gen_) {}

    vector<MutationType> ConvertToMutationType(
        const vector<IntegerT> &mutation_actions_as_ints)
    {
        vector<MutationType> mutation_actions;
        mutation_actions.reserve(mutation_actions_as_ints.size());
        for (const IntegerT action_as_int : mutation_actions_as_ints)
        {
            mutation_actions.push_back(static_cast<MutationType>(action_as_int));
        }
        return mutation_actions;
    }

    void Mutator::Mutate(shared_ptr<const Algorithm> *algorithm)
    {
        Mutate(1, algorithm);
    }

    void Mutator::Mutate(const IntegerT num_mutations, shared_ptr<const Algorithm> *algorithm)
    {
        if (mutate_prob_ >= 1.0 || rand_gen_->UniformProbability() < mutate_prob_)
        {
            auto mutated = make_unique<Algorithm>(**algorithm);
            for (IntegerT i = 0; i < num_mutations; ++i)
            {
                MutateImpl(mutated.get());
            }
            algorithm->reset(mutated.release());
        }
    }

    Mutator::Mutator()
        : allowed_actions_(ParseTextFormat<MutationTypeList>(
            "mutation_types: [ "
            "  ALTER_PARAM_MUTATION_TYPE, "
            "  RANDOMIZE_INSTRUCTION_MUTATION_TYPE, "
            "  RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE "
            "]")),
        mutate_prob_(0.5),
        allowed_setup_ops_(
            { NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP }),
        allowed_predict_ops_(
            { NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP }),
        allowed_learn_ops_(
            { NO_OP, SCALAR_SUM_OP, MATRIX_VECTOR_PRODUCT_OP, VECTOR_MEAN_OP }),
        mutate_setup_(!allowed_setup_ops_.empty()),
        mutate_predict_(!allowed_predict_ops_.empty()),
        mutate_learn_(!allowed_learn_ops_.empty()),
        setup_size_min_(2),
        setup_size_max_(4),
        predict_size_min_(3),
        predict_size_max_(5),
        learn_size_min_(4),
        learn_size_max_(6),
        bit_gen_owned_(make_unique<mt19937>(GenerateRandomSeed())),
        bit_gen_(bit_gen_owned_.get()),
        rand_gen_owned_(make_unique<RandomGenerator>(bit_gen_)),
        rand_gen_(rand_gen_owned_.get()),
        randomizer_(
            allowed_setup_ops_,
            allowed_predict_ops_,
            allowed_learn_ops_,
            bit_gen_,
            rand_gen_)
    {
    }

    void Mutator::MutateImpl(Algorithm *algorithm)
    {
        CHECK(!allowed_actions_.mutation_types().empty());
        const size_t action_index =
            absl::Uniform<size_t>(*bit_gen_, 0,
                allowed_actions_.mutation_types_size());
        const MutationType action = allowed_actions_.mutation_types(action_index);
        switch (action)
        {
        case ALTER_PARAM_MUTATION_TYPE:
            AlterParam(algorithm);
            return;
        case RANDOMIZE_INSTRUCTION_MUTATION_TYPE:
            RandomizeInstruction(algorithm);
            return;
        case RANDOMIZE_COMPONENT_FUNCTION_MUTATION_TYPE:
            RandomizeComponentFunction(algorithm);
            return;
        case IDENTITY_MUTATION_TYPE:
            return;
        case INSERT_INSTRUCTION_MUTATION_TYPE:
            InsertInstruction(algorithm);
            return;
        case REMOVE_INSTRUCTION_MUTATION_TYPE:
            RemoveInstruction(algorithm);
            return;
        case TRADE_INSTRUCTION_MUTATION_TYPE:
            TradeInstruction(algorithm);
            return;
        case RANDOMIZE_ALGORITHM_MUTATION_TYPE:
            RandomizeAlgorithm(algorithm);
            return;
            // Do not add a default clause here. All actions should be supported.
        }
    }

    void Mutator::AlterParam(Algorithm *algorithm)
    {
        ComponentFunction &componentFunction = getComponentFunction(algorithm, RandomComponentFunction());
        if (!componentFunction.instructions.empty())
        {
            InstructionIndexT index = RandomInstructionIndex(componentFunction.instructions.size());
            componentFunction.instructions[index] = make_shared<const Instruction>(*componentFunction.instructions[index], rand_gen_);
        }
    }

    void Mutator::RandomizeInstruction(Algorithm *algorithm)
    {
        ComponentFunctionT componentFunctionType = RandomComponentFunction();
        ComponentFunction &componentFunction = getComponentFunction(algorithm, componentFunctionType);
        if (!componentFunction.instructions.empty())
        {
            InstructionIndexT index = RandomInstructionIndex(componentFunction.instructions.size());
            componentFunction.instructions[index] = make_shared<const Instruction>(getRandomOp(componentFunctionType), rand_gen_);
        }
    }

    void Mutator::RandomizeComponentFunction(Algorithm *algorithm)
    {
        switch (RandomComponentFunction())
        {
        case kSetupComponentFunction:
        {
            randomizer_.RandomizeSetup(algorithm);
            return;
        }
        case kPredictComponentFunction:
        {
            randomizer_.RandomizePredict(algorithm);
            return;
        }
        case kLearnComponentFunction:
        {
            randomizer_.RandomizeLearn(algorithm);
            return;
        }
        }
        LOG(FATAL) << "Control flow should not reach here.";
    }

    void Mutator::InsertInstruction(Algorithm *algorithm)
    {
        ComponentFunctionT componentFunctionType = RandomComponentFunction();
        ComponentFunction &component_function = getComponentFunction(algorithm, componentFunctionType); // To modify.
        InstructionIndexT maxSize = getMaxSize(componentFunctionType);
        if (component_function.instructions.size() >= maxSize - 1)
            return;
        InsertInstructionUnconditionally(getRandomOp(componentFunctionType), &component_function.instructions);
    }

    InstructionIndexT Mutator::getMaxSize(ComponentFunctionT componentFunction) {
        switch (componentFunction)
        {
        case kSetupComponentFunction: return setup_size_max_;
        case kPredictComponentFunction: return predict_size_max_;
        case kLearnComponentFunction: return learn_size_max_;
        }
    }

    InstructionIndexT Mutator::getMinSize(ComponentFunctionT componentFunction) {
        switch (componentFunction)
        {
        case kSetupComponentFunction: return setup_size_min_;
        case kPredictComponentFunction: return predict_size_min_;
        case kLearnComponentFunction: return learn_size_min_;
        }
    }

    void Mutator::RemoveInstruction(Algorithm *algorithm)
    {
        ComponentFunctionT componentFunctionType = RandomComponentFunction();
        ComponentFunction &component_function = getComponentFunction(algorithm, componentFunctionType); // To modify.
        InstructionIndexT minSize = getMinSize(componentFunctionType);
        if (component_function.instructions.size() <= minSize)
            return;
        RemoveInstructionUnconditionally(&component_function.instructions);
    }

    void Mutator::TradeInstruction(Algorithm *algorithm)
    {
        ComponentFunctionT componentFunctionType = RandomComponentFunction();
        ComponentFunction &component_function = getComponentFunction(algorithm, componentFunctionType); // To modify.

        InsertInstructionUnconditionally(getRandomOp(componentFunctionType), &component_function.instructions);
        RemoveInstructionUnconditionally(&component_function.instructions);
    }

    void Mutator::RandomizeAlgorithm(Algorithm *algorithm)
    {
        if (mutate_setup_)
        {
            randomizer_.RandomizeSetup(algorithm);
        }
        if (mutate_predict_)
        {
            randomizer_.RandomizePredict(algorithm);
        }
        if (mutate_learn_)
        {
            randomizer_.RandomizeLearn(algorithm);
        }
    }

    void Mutator::InsertInstructionUnconditionally(
        const Op op, vector<shared_ptr<const Instruction>> *component_function)
    {
        const InstructionIndexT position = RandomInstructionIndex(component_function->size() + 1);
        component_function->insert(
            component_function->begin() + position,
            make_shared<const Instruction>(op, rand_gen_));
    }

    void Mutator::RemoveInstructionUnconditionally(
        vector<shared_ptr<const Instruction>> *component_function)
    {
        CHECK_GT(component_function->size(), 0);
        const InstructionIndexT position = RandomInstructionIndex(component_function->size());
        component_function->erase(component_function->begin() + position);
    }

    Op Mutator::RandomSetupOp()
    {
        return randomizer_.SetupOp();
    }

    Op Mutator::RandomPredictOp()
    {
        return randomizer_.PredictOp();
    }

    Op Mutator::RandomLearnOp()
    {
        return randomizer_.LearnOp();
    }

    Op Mutator::getRandomOp(ComponentFunctionT componentFunction) {
        switch (componentFunction) {
        case kSetupComponentFunction: return RandomSetupOp();
        case kPredictComponentFunction: return RandomPredictOp();
        case kLearnComponentFunction: return RandomLearnOp();
        default: LOG(FATAL) << "Control flow should not reach here.";
        }
    }

    InstructionIndexT Mutator::RandomInstructionIndex(const InstructionIndexT component_function_size)
    {
        return absl::Uniform<InstructionIndexT>(*bit_gen_, 0, component_function_size);
    }

    ComponentFunctionT Mutator::RandomComponentFunction()
    {
        vector<ComponentFunctionT> allowed_component_functions;
        allowed_component_functions.reserve(4);
        if (mutate_setup_)
        {
            allowed_component_functions.push_back(kSetupComponentFunction);
        }
        if (mutate_predict_)
        {
            allowed_component_functions.push_back(kPredictComponentFunction);
        }
        if (mutate_learn_)
        {
            allowed_component_functions.push_back(kLearnComponentFunction);
        }
        CHECK(!allowed_component_functions.empty())
            << "Must mutate at least one component function." << endl;
        const IntegerT index =
            absl::Uniform<IntegerT>(*bit_gen_, 0, allowed_component_functions.size());
        return allowed_component_functions[index];
    }

    ComponentFunction &Mutator::getComponentFunction(Algorithm *algorithm, ComponentFunctionT componentFunction) {
        switch (componentFunction)
        {
        case kSetupComponentFunction: return algorithm->setup_;
        case kPredictComponentFunction: return algorithm->predict_;
        case kLearnComponentFunction: return algorithm->learn_;
        default: LOG(FATAL) << "Control flow should not reach here.";
        }
    }
} // namespace automl_zero
