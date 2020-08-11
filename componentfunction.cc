#include "componentfunction.h"
#include "definitions.h"

namespace automl_zero {

    bool ComponentFunction::empty() const {
        return instructions.empty();
    }

    int ComponentFunction::size() const {
        return instructions.size();
    }

    void ComponentFunction::insert(const InstructionIndexT position, std::shared_ptr<const Instruction> instruction) {
        instructions.insert(
            instructions.begin() + position,
            instruction);
    }

    void ComponentFunction::remove(const InstructionIndexT position) {
        CHECK_GT(instructions.size(), 0);
        instructions.erase(instructions.begin() + position);
    }

    bool ComponentFunction::operator==(const ComponentFunction& other) const {
        const std::vector<std::shared_ptr<const Instruction>>& component_function1 = this->instructions;
        const std::vector<std::shared_ptr<const Instruction>>& component_function2 = other.instructions;
        if (component_function1.size() != component_function2.size()) {
            return false;
        }
        std::vector<std::shared_ptr<const Instruction>>::const_iterator instruction1_it = component_function1.begin();
        for (const std::shared_ptr<const Instruction>& instruction2 : component_function2) {
            if (*instruction2 != **instruction1_it) return false;
            ++instruction1_it;
        }
        CHECK(instruction1_it == component_function1.end());
        return true;
    }
}