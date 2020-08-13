#ifndef COMPONENTFUNCTION_H_
#define COMPONENTFUNCTION_H_

#include <array>
#include <memory>

#include "instruction.h"

namespace automl_zero
{

    class ComponentFunction
    {
    public:
        bool empty() const;
        int size() const;
        void insert(const InstructionIndexT position, std::shared_ptr<const Instruction> instruction);
        void remove(const InstructionIndexT position);
        bool operator==(const ComponentFunction &other) const;
        bool operator!=(const ComponentFunction &other) const
        {
            return !(*this == other);
        }
        void ShallowCopyTo(ComponentFunction &dest) const;
        std::vector<std::shared_ptr<const Instruction>> &getInstructions();
        const std::vector<std::shared_ptr<const Instruction>> &getConstInstructions() const;

    private:
        std::vector<std::shared_ptr<const Instruction>> instructions;
    };
} // namespace automl_zero

#endif
