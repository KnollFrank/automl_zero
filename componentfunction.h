#ifndef COMPONENTFUNCTION_H_
#define COMPONENTFUNCTION_H_

#include <array>
#include <memory>

#include "instruction.h"

namespace automl_zero {

    class ComponentFunction {
    public:
        bool empty() const;
        int size() const;
        void insert(const InstructionIndexT position, std::shared_ptr<const Instruction> instruction);
        std::vector<std::shared_ptr<const Instruction>> instructions;
    };
}

#endif
