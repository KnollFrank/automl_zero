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

#include "algorithm.h"

#include <sstream>
#include <string>
#include <vector>

#include "definitions.h"
#include "instruction.h"
#include "random_generator.h"
#include "absl/flags/flag.h"

namespace automl_zero {

    using ::std::istringstream;  // NOLINT
    using ::std::make_shared;  // NOLINT
    using ::std::ostream;  // NOLINT
    using ::std::ostringstream;  // NOLINT
    using ::std::shared_ptr;  // NOLINT
    using ::std::string;  // NOLINT
    using ::std::stringstream;  // NOLINT
    using ::std::vector;  // NOLINT

    Algorithm::Algorithm(const SerializedAlgorithm& checkpoint_algorithm) {
        this->FromProto(checkpoint_algorithm);
    }

    Algorithm::Algorithm(const Algorithm& other) {
        other.setup_.ShallowCopyTo(this->setup_);
        other.predict_.ShallowCopyTo(this->predict_);
        other.learn_.ShallowCopyTo(this->learn_);
    }

    Algorithm& Algorithm::operator=(const Algorithm& other) {
        if (&other != this) {
            other.setup_.ShallowCopyTo(this->setup_);
            other.predict_.ShallowCopyTo(this->predict_);
            other.learn_.ShallowCopyTo(this->learn_);
        }
        return *this;
    }

    Algorithm::Algorithm(Algorithm&& other) {
        setup_ = std::move(other.setup_);
        predict_ = std::move(other.predict_);
        learn_ = std::move(other.learn_);
    }

    Algorithm& Algorithm::operator=(Algorithm&& other) {
        if (&other != this) {
            setup_ = std::move(other.setup_);
            predict_ = std::move(other.predict_);
            learn_ = std::move(other.learn_);
        }
        return *this;
    }

    bool Algorithm::operator==(const Algorithm& other) const {
        return (setup_ == other.setup_) && (predict_ == other.predict_) && (learn_ == other.learn_);
    }

    string Algorithm::ToReadable() const {
        ostringstream stream;
        stream << "def Setup():" << std::endl;
        for (const shared_ptr<const Instruction>& instruction : setup_.getConstInstructions()) {
            stream << instruction->ToString();
        }
        stream << "def Predict():" << std::endl;
        for (const shared_ptr<const Instruction>& instruction : predict_.getConstInstructions()) {
            stream << instruction->ToString();
        }
        stream << "def Learn():" << std::endl;
        for (const shared_ptr<const Instruction>& instruction : learn_.getConstInstructions()) {
            stream << instruction->ToString();
        }
        return stream.str();
    }

    SerializedAlgorithm Algorithm::ToProto() const {
        SerializedAlgorithm checkpoint_algorithm;
        for (const shared_ptr<const Instruction>& instr : setup_.getConstInstructions()) {
            *checkpoint_algorithm.add_setup_instructions() = instr->Serialize();
        }
        for (const shared_ptr<const Instruction>& instr : predict_.getConstInstructions()) {
            *checkpoint_algorithm.add_predict_instructions() = instr->Serialize();
        }
        for (const shared_ptr<const Instruction>& instr : learn_.getConstInstructions()) {
            *checkpoint_algorithm.add_learn_instructions() = instr->Serialize();
        }
        return checkpoint_algorithm;
    }

    void Algorithm::FromProto(const SerializedAlgorithm& checkpoint_algorithm) {
        setup_.getInstructions().reserve(checkpoint_algorithm.setup_instructions_size());
        setup_.getInstructions().clear();
        for (const SerializedInstruction& checkpoint_instruction: checkpoint_algorithm.setup_instructions()) {
            setup_.getInstructions().emplace_back(
                make_shared<const Instruction>(checkpoint_instruction));
        }

        predict_.getInstructions().reserve(checkpoint_algorithm.predict_instructions_size());
        predict_.getInstructions().clear();
        for (const SerializedInstruction& checkpoint_instruction: checkpoint_algorithm.predict_instructions()) {
            predict_.getInstructions().emplace_back(
                make_shared<const Instruction>(checkpoint_instruction));
        }

        learn_.getInstructions().reserve(checkpoint_algorithm.learn_instructions_size());
        learn_.getInstructions().clear();
        for (const SerializedInstruction& checkpoint_instruction: checkpoint_algorithm.learn_instructions()) {
            learn_.getInstructions().emplace_back(
                make_shared<const Instruction>(checkpoint_instruction));
        }
    }

    const ComponentFunction& Algorithm::ComponentFunction(const ComponentFunctionT component_function_type) const {
        switch (component_function_type) {
        case kSetupComponentFunction:
            return setup_;
        case kPredictComponentFunction:
            return predict_;
        case kLearnComponentFunction:
            return learn_;
        }
    }

    ComponentFunction* Algorithm::MutableComponentFunction(const ComponentFunctionT component_function_type) {
        switch (component_function_type) {
        case kSetupComponentFunction:
            return &setup_;
        case kPredictComponentFunction:
            return &predict_;
        case kLearnComponentFunction:
            return &learn_;
        }
    }

}  // namespace automl_zero
