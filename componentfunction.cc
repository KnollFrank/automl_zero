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
}